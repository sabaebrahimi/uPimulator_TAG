package program

import (
	"os"
	"path/filepath"
	"testing"
)

// writeLines writes one string per line (the mram.bin / addresses.txt format).
func writeLines(t *testing.T, path string, lines []string) {
	t.Helper()
	data := ""
	for _, l := range lines {
		data += l + "\n"
	}
	if err := os.WriteFile(path, []byte(data), 0644); err != nil {
		t.Fatal(err)
	}
}

// TestPimdlMramPatchSplicesSegments verifies the core patch behavior:
//   - manifest + segments are read from patchDirpath (separate from binDirpath),
//   - a named anchor resolves the base VA,
//   - each segment lands at (symbolVA - anchorVA) in mram.bin with exact bytes.
func TestPimdlMramPatchSplicesSegments(t *testing.T) {
	binDir := t.TempDir()
	patchDir := t.TempDir()

	// mram.bin: 200 zero bytes (one decimal per line).
	mram := make([]string, 200)
	for i := range mram {
		mram[i] = "0"
	}
	writeLines(t, filepath.Join(binDir, "mram.bin"), mram)

	// addresses.txt: a synthetic base anchor at VA 0 and two named MRAM symbols.
	writeLines(t, filepath.Join(binDir, "addresses.txt"), []string{
		"BASE: 0",
		"lut_table: 16",
		"input_index: 48",
	})

	// segment payloads live in the patch dir (NOT the wiped bin dir).
	segDir := filepath.Join(patchDir, "pimdl_segments")
	if err := os.MkdirAll(segDir, 0755); err != nil {
		t.Fatal(err)
	}
	lutBytes := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	idxBytes := []byte{9, 10, 11, 12}
	if err := os.WriteFile(filepath.Join(segDir, "lut_table.bin"), lutBytes, 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(segDir, "input_index.bin"), idxBytes, 0644); err != nil {
		t.Fatal(err)
	}

	manifest := `{
	  "anchor": "BASE",
	  "segments": [
	    {"symbol": "lut_table",   "path": "pimdl_segments/lut_table.bin"},
	    {"symbol": "input_index", "path": "pimdl_segments/input_index.bin"}
	  ]
	}`
	if err := os.WriteFile(filepath.Join(patchDir, pimdlMRAMManifestName), []byte(manifest), 0644); err != nil {
		t.Fatal(err)
	}

	RunPimdlMramPatchIfPresent(binDir, patchDir)

	got, err := readPimdlMRAMLines(filepath.Join(binDir, "mram.bin"))
	if err != nil {
		t.Fatal(err)
	}

	check := func(start int, payload []byte) {
		for i, b := range payload {
			want := itoa(int(b))
			if got[start+i] != want {
				t.Fatalf("mram.bin[%d] = %q, want %q", start+i, got[start+i], want)
			}
		}
	}
	// lut_table at VA 16 - BASE 0 = offset 16; input_index at 48.
	check(16, lutBytes)
	check(48, idxBytes)

	// bytes outside the patched regions stay zero.
	if got[0] != "0" || got[15] != "0" || got[24] != "0" || got[199] != "0" {
		t.Fatalf("unexpected mutation outside patched regions")
	}
}

// TestPimdlMramPatchNoManifestIsNoop ensures absence of a manifest is a silent no-op.
func TestPimdlMramPatchNoManifestIsNoop(t *testing.T) {
	binDir := t.TempDir()
	patchDir := t.TempDir()
	writeLines(t, filepath.Join(binDir, "mram.bin"), []string{"7", "7", "7"})

	RunPimdlMramPatchIfPresent(binDir, patchDir) // must not panic

	got, err := readPimdlMRAMLines(filepath.Join(binDir, "mram.bin"))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0] != "7" {
		t.Fatalf("mram.bin changed by no-op patch: %v", got)
	}
}

func itoa(v int) string {
	// tiny local helper to avoid importing strconv in the test for one call
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}
	buf := [12]byte{}
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
