package vm

import (
	"fmt"
	"os"
	"path/filepath"

	"uPIMulator/src/encoding"
	"uPIMulator/src/host/vm/dram/bank"
)

// dpuCoords maps a flat DPU index (order of VirtualMachine.Dpus()) onto the
// (channel, rank, dpu) triple expected by TransferCommand / ChannelCommand.
func (this *VirtualMachine) dpuCoords(flat_id int) (channel_id int, rank_id int, dpu_id int) {
	per_channel := this.num_ranks_per_channel * this.num_dpus_per_rank
	channel_id = flat_id / per_channel
	rank_id = (flat_id % per_channel) / this.num_dpus_per_rank
	dpu_id = flat_id % this.num_dpus_per_rank
	return
}

// enqueueMramTransfer builds a TransferCommand and registers it in push_xfer so
// SimulateMemory will drive it through the host↔MRAM timing model.
func (this *VirtualMachine) enqueueMramTransfer(
	xfer_type bank.TransferCommandType,
	vm_address int64,
	channel_id int,
	rank_id int,
	dpu_id int,
	mram_address int64,
	size int64,
) {
	transfer_command := new(bank.TransferCommand)
	transfer_command.Init(
		xfer_type,
		vm_address,
		channel_id,
		rank_id,
		dpu_id,
		mram_address,
		size,
	)
	this.memory_controller.Push(transfer_command)
	this.push_xfer[transfer_command] = true
}

// SimulatePimdlHostToDeviceTransfers charges HOST_TO_DEVICE cycles for the
// co-sim LUT/index payloads. Active only when PIMDL_OUTPUT_BYTES is set (same
// gate as DumpPimdlOutput) and a patch directory with segment files exists.
//
// The mram-patch has already placed the correct bytes in MRAM before DpuLoad;
// this path re-drives those same bytes through the timed transfer model so
// co-sim reports the host→DPU data-transfer overhead that the dump-based
// handoff previously skipped.
func (this *VirtualMachine) SimulatePimdlHostToDeviceTransfers() {
	if os.Getenv("PIMDL_OUTPUT_BYTES") == "" {
		return
	}
	if this.pimdl_patch_dirpath == "" {
		return
	}

	segments := []struct {
		symbol string
		rel    string
	}{
		{"lut_table", filepath.Join("pimdl_segments", "lut_table.bin")},
		{"input_index", filepath.Join("pimdl_segments", "input_index.bin")},
	}

	type staged struct {
		symbol  string
		va      int64
		payload []byte
		host_va int64
	}
	staged_segs := make([]staged, 0, len(segments))

	for _, seg := range segments {
		va, found := this.task.Addresses()[seg.symbol]
		if !found {
			fmt.Printf("pimdl: timed H2D skipped for %s (symbol missing)\n", seg.symbol)
			continue
		}
		path := filepath.Join(this.pimdl_patch_dirpath, seg.rel)
		payload, err := os.ReadFile(path)
		if err != nil {
			fmt.Printf("pimdl: timed H2D skipped for %s (%v)\n", seg.symbol, err)
			continue
		}
		if len(payload) == 0 {
			continue
		}

		host_buf := this.arena.NewPointer(int64(len(payload)))
		byte_stream := new(encoding.ByteStream)
		byte_stream.Init()
		for _, b := range payload {
			byte_stream.Append(b)
		}
		this.arena.Pool().Memory().Write(host_buf.Address(), int64(len(payload)), byte_stream)

		staged_segs = append(staged_segs, staged{
			symbol:  seg.symbol,
			va:      va,
			payload: payload,
			host_va: host_buf.Address(),
		})
	}

	if len(staged_segs) == 0 {
		return
	}

	this.Checkpoint()

	var total_bytes int64
	for _, seg := range staged_segs {
		for flat_id := range this.Dpus() {
			channel_id, rank_id, dpu_id := this.dpuCoords(flat_id)
			this.enqueueMramTransfer(
				bank.HOST_TO_DEVICE,
				seg.host_va,
				channel_id,
				rank_id,
				dpu_id,
				seg.va,
				int64(len(seg.payload)),
			)
			total_bytes += int64(len(seg.payload))
		}
		this.stat_factory.Increment("num_h2d", int64(len(this.Dpus())))
	}

	cycles := this.SimulateMemory()
	this.stat_factory.Increment("h2d_cycle", cycles)
	this.stat_factory.Increment("h2d_bytes", total_bytes)

	fmt.Printf("pimdl: timed H2D transfer %d bytes across %d segment(s) (h2d_cycle=%d)\n",
		total_bytes, len(staged_segs), cycles)
}
