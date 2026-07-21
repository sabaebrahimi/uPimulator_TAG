package vm

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"uPIMulator/src/encoding"
	"uPIMulator/src/host/vm/dram/bank"
)

// Paper / upmem_reg_model CPU↔DPU channel bandwidths (GB/s per DPU).
// Parallel push_xfer wall time ≈ per-DPU shard size / BW (equal shards).
const (
	pimdlH2DBandwidthGBps = 0.2957
	pimdlD2HBandwidthGBps = 0.0627
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

// pimdlXferMode returns "fixed_bw" (default, safe for multi-DPU) or "cycle".
// Cycle-accurate SimulateMemory for many DPUs can OOM the host (thread-pool
// per cycle × DPU count × transfer length); refuse that unless explicitly
// forced and only recommend it for 1 DPU.
func (this *VirtualMachine) pimdlXferMode() string {
	mode := strings.ToLower(strings.TrimSpace(os.Getenv("COSIM_XFER_MODE")))
	if mode == "" {
		mode = "fixed_bw"
	}
	if mode == "cycle" && len(this.Dpus()) > 1 {
		fmt.Printf("pimdl: COSIM_XFER_MODE=cycle is unsafe with %d DPUs "+
			"(can exhaust RAM); falling back to fixed_bw. "+
			"Use 1 DPU for cycle-accurate xfer.\n", len(this.Dpus()))
		return "fixed_bw"
	}
	return mode
}

// bytesToMemCycles converts a wall-time from the fixed-BW model into memory
// cycles at the configured DPU MRAM / transfer clock.
func (this *VirtualMachine) bytesToMemCycles(nbytes int64, gbps float64) int64 {
	if nbytes <= 0 || gbps <= 0 || this.memory_frequency_mhz <= 0 {
		return 0
	}
	// t_sec = nbytes / (gbps * 2^30); cycles = t_sec * mhz * 1e6
	secs := float64(nbytes) / (gbps * float64(int64(1)<<30))
	return int64(secs * float64(this.memory_frequency_mhz) * 1e6)
}

// chargeFixedBandwidthTransfer records HostTransfer_* stats using the classic
// uPIMulator size/BW model (HPCA'24 Table I). No SimulateMemory.
func (this *VirtualMachine) chargeFixedBandwidthTransfer(
	direction string, // "h2d" or "d2h"
	per_dpu_bytes int64,
	num_dpus int,
) {
	if per_dpu_bytes <= 0 || num_dpus <= 0 {
		return
	}

	var gbps float64
	if direction == "h2d" {
		gbps = pimdlH2DBandwidthGBps
	} else {
		gbps = pimdlD2HBandwidthGBps
	}

	// Parallel xfer: wall ≈ per-DPU size / BW (all DPUs same shard size).
	cycles := this.bytesToMemCycles(per_dpu_bytes, gbps)
	total_bytes := per_dpu_bytes * int64(num_dpus)

	this.stat_factory.Increment("transfer_cycle", cycles)
	if direction == "h2d" {
		this.stat_factory.Increment("h2d_cycle", cycles)
		this.stat_factory.Increment("h2d_bytes", total_bytes)
		this.stat_factory.Increment("num_h2d", int64(num_dpus))
	} else {
		this.stat_factory.Increment("d2h_cycle", cycles)
		this.stat_factory.Increment("d2h_bytes", total_bytes)
		this.stat_factory.Increment("num_d2h", int64(num_dpus))
	}
	this.stat_factory.Increment("xfer_model_fixed_bw", 1)

	fmt.Printf("pimdl: fixed-BW %s: %d B/DPU × %d DPUs -> wall_cycle=%d "+
		"(bw=%.4f GB/s, total_bytes=%d)\n",
		direction, per_dpu_bytes, num_dpus, cycles, gbps, total_bytes)
}

// SimulatePimdlHostToDeviceTransfers accounts for HOST_TO_DEVICE of lut/index.
// Default: paper fixed-BW model (safe). Optional COSIM_XFER_MODE=cycle uses
// SimulateMemory but only for 1 DPU.
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

	var per_dpu_bytes int64
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
			fmt.Printf("pimdl: H2D skipped for %s (symbol missing)\n", seg.symbol)
			continue
		}
		path := filepath.Join(this.pimdl_patch_dirpath, seg.rel)
		payload, err := os.ReadFile(path)
		if err != nil {
			fmt.Printf("pimdl: H2D skipped for %s (%v)\n", seg.symbol, err)
			continue
		}
		if len(payload) == 0 {
			continue
		}
		per_dpu_bytes += int64(len(payload))

		if this.pimdlXferMode() == "cycle" {
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
	}

	if per_dpu_bytes == 0 {
		return
	}

	num_dpus := len(this.Dpus())
	if num_dpus == 0 {
		return
	}

	if this.pimdlXferMode() != "cycle" {
		this.chargeFixedBandwidthTransfer("h2d", per_dpu_bytes, num_dpus)
		return
	}

	// 1-DPU cycle-accurate path only.
	this.Checkpoint()
	var total_bytes int64
	for _, seg := range staged_segs {
		channel_id, rank_id, dpu_id := this.dpuCoords(0)
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
		this.stat_factory.Increment("num_h2d", 1)
	}
	cycles := this.SimulateMemory()
	this.stat_factory.Increment("h2d_cycle", cycles)
	this.stat_factory.Increment("h2d_bytes", total_bytes)
	this.stat_factory.Increment("xfer_model_cycle", 1)
	fmt.Printf("pimdl: cycle H2D transfer %d bytes (h2d_cycle=%d)\n", total_bytes, cycles)
}
