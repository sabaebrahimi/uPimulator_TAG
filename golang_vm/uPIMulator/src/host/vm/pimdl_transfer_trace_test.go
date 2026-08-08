package vm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeTransferTrace(t *testing.T, lines ...string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "trace.jsonl")
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadPimdlTransferTraceGroupsBatches(t *testing.T) {
	trace, err := loadPimdlTransferTrace(writeTransferTrace(t,
		`{"version":1,"type":"header","num_dpus":2}`,
		`{"version":1,"type":"transfer","boundary":"qkv","batch":0,"direction":"h2d","dpu":0,"symbol":"input","mram_offset":0,"bytes":8,"host_offset":0,"kind":"input"}`,
		`{"version":1,"type":"transfer","boundary":"qkv","batch":0,"direction":"h2d","dpu":1,"symbol":"input","mram_offset":8,"bytes":8,"host_offset":8,"kind":"input"}`,
		`{"version":1,"type":"transfer","boundary":"qkv","batch":1,"direction":"d2h","dpu":0,"symbol":"output","mram_offset":0,"bytes":4,"host_offset":0,"kind":"output"}`,
	))
	if err != nil {
		t.Fatal(err)
	}
	if trace.NumDpus != 2 || trace.Boundary != "qkv" || len(trace.Batches) != 2 {
		t.Fatalf("unexpected trace shape: %+v", trace)
	}
	if trace.Batches[0].Direction != "h2d" || len(trace.Batches[0].Events) != 2 {
		t.Fatalf("unexpected first batch: %+v", trace.Batches[0])
	}
	if err := validatePimdlTransferTrace(
		trace, 2, map[string]int64{"input": 512, "output": 544}, 512, 1024,
	); err != nil {
		t.Fatalf("validate trace: %v", err)
	}
	if trace.Batches[0].Events[1].mramVA != 520 {
		t.Fatalf("resolved MRAM VA = %d, want 520", trace.Batches[0].Events[1].mramVA)
	}
}

func TestLoadPimdlTransferTraceAllowsDirectionChangesForOneBoundary(t *testing.T) {
	trace, err := loadPimdlTransferTrace(writeTransferTrace(t,
		`{"version":1,"type":"header","num_dpus":1}`,
		`{"version":1,"type":"transfer","boundary":"qkv","batch":0,"direction":"d2h","dpu":0,"symbol":"output","mram_offset":0,"bytes":4,"host_offset":0,"kind":"output"}`,
		`{"version":1,"type":"transfer","boundary":"qkv","batch":1,"direction":"h2d","dpu":0,"symbol":"input","mram_offset":0,"bytes":4,"host_offset":0,"kind":"input"}`,
	))
	if err != nil {
		t.Fatalf("direction changes within one boundary must be accepted: %v", err)
	}
	if trace.Boundary != "qkv" {
		t.Fatalf("trace boundary = %q, want qkv", trace.Boundary)
	}
}

func TestLoadPimdlTransferTraceRejectsMixedBoundaries(t *testing.T) {
	_, err := loadPimdlTransferTrace(writeTransferTrace(t,
		`{"version":1,"type":"header","num_dpus":1}`,
		`{"version":1,"type":"transfer","boundary":"qkv","batch":0,"direction":"d2h","dpu":0,"symbol":"output","mram_offset":0,"bytes":4,"host_offset":0,"kind":"output"}`,
		`{"version":1,"type":"transfer","boundary":"o","batch":1,"direction":"h2d","dpu":0,"symbol":"input","mram_offset":0,"bytes":4,"host_offset":0,"kind":"input"}`,
	))
	if err == nil || !strings.Contains(err.Error(), "differs from trace boundary") {
		t.Fatalf("mixed-boundary error = %v", err)
	}
}

func TestLoadPimdlTransferTraceRejectsSchemaAndTopology(t *testing.T) {
	_, err := loadPimdlTransferTrace(writeTransferTrace(t,
		`{"version":1,"type":"header","num_dpus":1,"extra":true}`,
	))
	if err == nil {
		t.Fatal("expected unknown header field to fail")
	}

	_, err = loadPimdlTransferTrace(writeTransferTrace(t,
		`{"version":1,"type":"header","num_dpus":1}`,
		`{"version":1,"type":"transfer","boundary":"qkv","batch":0,"direction":"h2d","dpu":1,"symbol":"input","mram_offset":0,"bytes":4,"host_offset":0,"kind":"input"}`,
	))
	if err == nil || !strings.Contains(err.Error(), "outside header topology") {
		t.Fatalf("topology error = %v", err)
	}
}

func TestValidatePimdlTransferTraceRejectsSymbolAndBounds(t *testing.T) {
	trace, err := loadPimdlTransferTrace(writeTransferTrace(t,
		`{"version":1,"type":"header","num_dpus":1}`,
		`{"version":1,"type":"transfer","boundary":"qkv","batch":0,"direction":"h2d","dpu":0,"symbol":"input","mram_offset":12,"bytes":8,"host_offset":0,"kind":"input"}`,
	))
	if err != nil {
		t.Fatal(err)
	}
	if err := validatePimdlTransferTrace(trace, 1, map[string]int64{}, 512, 528); err == nil ||
		!strings.Contains(err.Error(), "not in task addresses") {
		t.Fatalf("missing symbol error = %v", err)
	}
	if err := validatePimdlTransferTrace(trace, 1, map[string]int64{"input": 512}, 512, 528); err == nil ||
		!strings.Contains(err.Error(), "outside MRAM") {
		t.Fatalf("bounds error = %v", err)
	}
}
