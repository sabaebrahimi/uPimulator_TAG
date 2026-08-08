package vm

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"uPIMulator/src/host/vm/dram/bank"
	"uPIMulator/src/misc"
)

const pimdlTransferTraceVersion = 1

type pimdlTransferTraceEvent struct {
	Boundary   string
	Batch      int64
	Direction  string
	Dpu        int64
	Symbol     string
	MramOffset int64
	Bytes      int64
	HostOffset int64
	Kind       string
	mramVA     int64
}

type pimdlTransferTraceBatch struct {
	ID        int64
	Boundary  string
	Direction string
	Events    []pimdlTransferTraceEvent
}

type pimdlTransferTrace struct {
	NumDpus  int64
	Boundary string
	Batches  []pimdlTransferTraceBatch
}

type pimdlTransferTraceHeaderLine struct {
	Version *int64  `json:"version"`
	Type    *string `json:"type"`
	NumDpus *int64  `json:"num_dpus"`
}

type pimdlTransferTraceEventLine struct {
	Version    *int64  `json:"version"`
	Type       *string `json:"type"`
	Boundary   *string `json:"boundary"`
	Batch      *int64  `json:"batch"`
	Direction  *string `json:"direction"`
	Dpu        *int64  `json:"dpu"`
	Symbol     *string `json:"symbol"`
	MramOffset *int64  `json:"mram_offset"`
	Bytes      *int64  `json:"bytes"`
	HostOffset *int64  `json:"host_offset"`
	Kind       *string `json:"kind"`
}

func decodePimdlTransferTraceLine(raw []byte, destination interface{}) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	var extra interface{}
	if err := decoder.Decode(&extra); err != io.EOF {
		return errors.New("contains more than one JSON value")
	}
	return nil
}

func loadPimdlTransferTrace(path string) (*pimdlTransferTrace, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 4096), 4*1024*1024)
	trace := new(pimdlTransferTrace)
	lineNumber := 0
	var currentBatch *pimdlTransferTraceBatch
	lastBatch := int64(-1)
	for scanner.Scan() {
		lineNumber++
		raw := bytes.TrimSpace(scanner.Bytes())
		if len(raw) == 0 {
			return nil, fmt.Errorf("pimdl transfer trace line %d: blank lines are not allowed", lineNumber)
		}

		if lineNumber == 1 {
			var header pimdlTransferTraceHeaderLine
			if err := decodePimdlTransferTraceLine(raw, &header); err != nil {
				return nil, fmt.Errorf("pimdl transfer trace header: %w", err)
			}
			if header.Version == nil || *header.Version != pimdlTransferTraceVersion ||
				header.Type == nil || *header.Type != "header" ||
				header.NumDpus == nil || *header.NumDpus <= 0 {
				return nil, errors.New("pimdl transfer trace: invalid version-1 header")
			}
			trace.NumDpus = *header.NumDpus
			continue
		}

		var eventLine pimdlTransferTraceEventLine
		if err := decodePimdlTransferTraceLine(raw, &eventLine); err != nil {
			return nil, fmt.Errorf("pimdl transfer trace line %d: %w", lineNumber, err)
		}
		if eventLine.Version == nil || *eventLine.Version != pimdlTransferTraceVersion ||
			eventLine.Type == nil || *eventLine.Type != "transfer" ||
			eventLine.Boundary == nil || *eventLine.Boundary == "" ||
			eventLine.Batch == nil || *eventLine.Batch < 0 ||
			eventLine.Direction == nil || (*eventLine.Direction != "h2d" && *eventLine.Direction != "d2h") ||
			eventLine.Dpu == nil || *eventLine.Dpu < 0 ||
			eventLine.Symbol == nil || *eventLine.Symbol == "" ||
			eventLine.MramOffset == nil || *eventLine.MramOffset < 0 ||
			eventLine.Bytes == nil || *eventLine.Bytes <= 0 ||
			eventLine.HostOffset == nil || *eventLine.HostOffset < 0 ||
			eventLine.Kind == nil || *eventLine.Kind == "" {
			return nil, fmt.Errorf("pimdl transfer trace line %d: invalid transfer event", lineNumber)
		}
		if *eventLine.Dpu >= trace.NumDpus {
			return nil, fmt.Errorf("pimdl transfer trace line %d: dpu %d outside header topology",
				lineNumber, *eventLine.Dpu)
		}
		if *eventLine.Batch < lastBatch {
			return nil, fmt.Errorf("pimdl transfer trace line %d: batch %d follows batch %d",
				lineNumber, *eventLine.Batch, lastBatch)
		}
		if trace.Boundary == "" {
			trace.Boundary = *eventLine.Boundary
		} else if *eventLine.Boundary != trace.Boundary {
			return nil, fmt.Errorf("pimdl transfer trace line %d: boundary %q differs from trace boundary %q",
				lineNumber, *eventLine.Boundary, trace.Boundary)
		}

		event := pimdlTransferTraceEvent{
			Boundary:   *eventLine.Boundary,
			Batch:      *eventLine.Batch,
			Direction:  *eventLine.Direction,
			Dpu:        *eventLine.Dpu,
			Symbol:     *eventLine.Symbol,
			MramOffset: *eventLine.MramOffset,
			Bytes:      *eventLine.Bytes,
			HostOffset: *eventLine.HostOffset,
			Kind:       *eventLine.Kind,
		}
		if currentBatch == nil || currentBatch.ID != event.Batch {
			trace.Batches = append(trace.Batches, pimdlTransferTraceBatch{
				ID:        event.Batch,
				Boundary:  event.Boundary,
				Direction: event.Direction,
			})
			currentBatch = &trace.Batches[len(trace.Batches)-1]
			lastBatch = event.Batch
		} else if currentBatch.Direction != event.Direction || currentBatch.Boundary != event.Boundary {
			return nil, fmt.Errorf("pimdl transfer trace line %d: mixed boundary or direction in batch %d",
				lineNumber, event.Batch)
		}
		currentBatch.Events = append(currentBatch.Events, event)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if lineNumber == 0 {
		return nil, errors.New("pimdl transfer trace: missing header")
	}
	if len(trace.Batches) == 0 {
		return nil, errors.New("pimdl transfer trace: contains no transfer events")
	}
	return trace, nil
}

func (this *VirtualMachine) transferTracePath() string {
	if this.pimdl_xfer_trace_path != "" {
		return this.pimdl_xfer_trace_path
	}
	if this.pimdl_patch_dirpath == "" {
		return ""
	}
	if boundary := this.pimdlTraceBoundary(); boundary != "" {
		return filepath.Join(this.pimdl_patch_dirpath, "pimdl_xfer_trace_"+boundary+".jsonl")
	}
	return filepath.Join(this.pimdl_patch_dirpath, "pimdl_xfer_trace.jsonl")
}

func (this *VirtualMachine) pimdlTraceBoundary() string {
	if this.task == nil {
		return ""
	}

	switch this.task.Benchmark() {
	case "PIMDL":
		return "qkv"
	case "PIMDL_O":
		return "o"
	case "PIMDL_FFN1":
		return "ffn1"
	case "PIMDL_FFN2":
		return "ffn2"
	default:
		return ""
	}
}

func (this *VirtualMachine) loadPimdlTransferTraceIfPresent() (*pimdlTransferTrace, bool) {
	if this.pimdlXferMode() != "trace_cycle" {
		return nil, false
	}
	if this.pimdl_trace_checked {
		return this.pimdl_transfer_trace, this.pimdl_transfer_trace != nil
	}
	this.pimdl_trace_checked = true

	path := this.transferTracePath()
	if path == "" {
		return nil, false
	}
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		fmt.Printf("pimdl: transfer trace %s unavailable; using existing transfer behavior\n", path)
		return nil, false
	} else if err != nil {
		panic(fmt.Errorf("pimdl transfer trace: %w", err))
	}

	trace, err := loadPimdlTransferTrace(path)
	if err != nil {
		panic(err)
	}
	configLoader := new(misc.ConfigLoader)
	configLoader.Init()
	if err := validatePimdlTransferTrace(
		trace,
		int64(len(this.Dpus())),
		this.task.Addresses(),
		configLoader.MramOffset(),
		configLoader.MramOffset()+configLoader.MramSize(),
	); err != nil {
		panic(err)
	}
	this.pimdl_transfer_trace = trace
	fmt.Printf("pimdl: loaded transfer trace %s (%d batches)\n", path, len(trace.Batches))
	return trace, true
}

func validatePimdlTransferTrace(
	trace *pimdlTransferTrace,
	numDpus int64,
	addresses map[string]int64,
	mramStart int64,
	mramEnd int64,
) error {
	if trace.NumDpus != numDpus {
		return fmt.Errorf("pimdl transfer trace: header has %d DPUs, simulator has %d",
			trace.NumDpus, numDpus)
	}
	if trace.Boundary == "" {
		return errors.New("pimdl transfer trace: boundary is empty")
	}
	for batchIndex := range trace.Batches {
		if trace.Batches[batchIndex].Boundary != trace.Boundary {
			return fmt.Errorf("pimdl transfer trace: batch %d boundary %q differs from trace boundary %q",
				trace.Batches[batchIndex].ID, trace.Batches[batchIndex].Boundary, trace.Boundary)
		}
		for eventIndex := range trace.Batches[batchIndex].Events {
			event := &trace.Batches[batchIndex].Events[eventIndex]
			if event.Boundary != trace.Boundary {
				return fmt.Errorf("pimdl transfer trace: batch %d event boundary %q differs from trace boundary %q",
					trace.Batches[batchIndex].ID, event.Boundary, trace.Boundary)
			}
			if event.Dpu < 0 || event.Dpu >= numDpus {
				return fmt.Errorf("pimdl transfer trace: dpu %d is outside simulator topology", event.Dpu)
			}
			symbolVA, found := addresses[event.Symbol]
			if !found {
				return fmt.Errorf("pimdl transfer trace: symbol %q is not in task addresses", event.Symbol)
			}
			if symbolVA > 0 && event.MramOffset > math.MaxInt64-symbolVA {
				return fmt.Errorf("pimdl transfer trace: %s MRAM offset overflows", event.Symbol)
			}
			event.mramVA = symbolVA + event.MramOffset
			if event.mramVA < mramStart || event.mramVA > mramEnd ||
				event.Bytes > mramEnd-event.mramVA {
				return fmt.Errorf("pimdl transfer trace: %s offset=%d bytes=%d is outside MRAM",
					event.Symbol, event.MramOffset, event.Bytes)
			}
			if event.HostOffset > math.MaxInt64-event.Bytes {
				return fmt.Errorf("pimdl transfer trace: host offset overflows for %s", event.Symbol)
			}
		}
	}
	return nil
}

func (this *VirtualMachine) replayPimdlTransferTrace(direction string) bool {
	trace, found := this.loadPimdlTransferTraceIfPresent()
	if !found {
		return false
	}

	replayed := false
	for _, batch := range trace.Batches {
		if batch.Direction != direction {
			continue
		}
		replayed = true

		type stagedTransfer struct {
			event  pimdlTransferTraceEvent
			hostVA int64
		}
		var arenaBytes int64
		for _, event := range batch.Events {
			end := event.HostOffset + event.Bytes
			if end > arenaBytes {
				arenaBytes = end
			}
		}
		allocation := this.arena.NewPointer(arenaBytes)
		staged := make([]stagedTransfer, 0, len(batch.Events))
		for _, event := range batch.Events {
			hostVA := allocation.Address() + event.HostOffset
			if direction == "h2d" {
				payload := this.Dpus()[event.Dpu].Dma().TransferFromMram(event.mramVA, event.Bytes)
				this.arena.Pool().Memory().Write(hostVA, event.Bytes, payload)
			}
			staged = append(staged, stagedTransfer{event: event, hostVA: hostVA})
		}
		if direction == "h2d" {
			// The controller obtains all host payloads from its VM memory during
			// the transfer, so checkpoint once after staging the complete batch.
			this.Checkpoint()
		}

		var transferType bank.TransferCommandType
		if direction == "h2d" {
			transferType = bank.HOST_TO_DEVICE
		} else {
			transferType = bank.DEVICE_TO_HOST
		}
		var totalBytes int64
		for _, transfer := range staged {
			channelID, rankID, dpuID := this.dpuCoords(int(transfer.event.Dpu))
			this.enqueueMramTransfer(
				transferType,
				transfer.hostVA,
				channelID,
				rankID,
				dpuID,
				transfer.event.mramVA,
				transfer.event.Bytes,
			)
			totalBytes += transfer.event.Bytes
		}
		cycles := this.SimulateMemory()
		this.stat_factory.Increment("xfer_model_trace_cycle", 1)
		if direction == "h2d" {
			this.stat_factory.Increment("h2d_cycle", cycles)
			this.stat_factory.Increment("h2d_bytes", totalBytes)
			this.stat_factory.Increment("num_h2d", int64(len(staged)))
		} else {
			this.stat_factory.Increment("d2h_cycle", cycles)
			this.stat_factory.Increment("d2h_bytes", totalBytes)
			this.stat_factory.Increment("num_d2h", int64(len(staged)))
		}
		fmt.Printf("pimdl: trace-cycle %s batch=%d events=%d bytes=%d cycles=%d\n",
			direction, batch.ID, len(staged), totalBytes, cycles)
	}
	return replayed
}
