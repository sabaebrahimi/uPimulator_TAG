package program

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// pimdlMRAMManifestName, if present under bin_dirpath after link, triggers a
// segment-oriented patch of mram.bin (no full-image precompute). PIM-DL (or
// any external tool) writes raw segment files; this hook splices them into
// the linker-produced mram.bin ASCII format (one decimal byte per line).
//
// Manifest schema (JSON):
//
//	{
//	  "anchor": "DPU_MRAM_HEAP_POINTER",
//	  "segments": [
//	    { "symbol": "lut_table", "path": "pimdl_segments/lut_table.bin" },
//	    { "path": "pimdl_segments/footer.bin", "offset_bytes": 400 }
//	  ]
//	}
//
// For each segment, either set "offset_bytes" (byte index into mram.bin) or
// set "symbol" so the start index is addresses[symbol] - addresses[anchor].
// That relative layout matches the usual case where mram.bin begins at the
// heap anchor and MRAM sections are contiguous from there; if your link map
// differs, use explicit offset_bytes only.
const pimdlMRAMManifestName = "pimdl_mram_patch.json"

type pimdlMRAMManifest struct {
	Anchor   string           `json:"anchor"`
	Segments []pimdlMRAMPatch `json:"segments"`
}

type pimdlMRAMPatch struct {
	Symbol      string `json:"symbol"`
	OffsetBytes *int64 `json:"offset_bytes"`
	Path        string `json:"path"`
}

func RunPimdlMramPatchIfPresent(binDirpath string) {
	manifestPath := filepath.Join(binDirpath, pimdlMRAMManifestName)
	if _, err := os.Stat(manifestPath); err != nil {
		return
	}

	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		panic(err)
	}

	var manifest pimdlMRAMManifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		panic(err)
	}
	if manifest.Anchor == "" {
		manifest.Anchor = "DPU_MRAM_HEAP_POINTER"
	}

	addressesPath := filepath.Join(binDirpath, "addresses.txt")
	addresses := parsePimdlAddresses(addressesPath)

	mramPath := filepath.Join(binDirpath, "mram.bin")
	lines, err := readPimdlMRAMLines(mramPath)
	if err != nil {
		panic(err)
	}

	anchorVA, ok := addresses[manifest.Anchor]
	if !ok {
		panic(fmt.Errorf(
			"pimdl mram patch: anchor %q not found in addresses.txt",
			manifest.Anchor,
		))
	}

	for _, seg := range manifest.Segments {
		if seg.Path == "" {
			panic(errors.New("pimdl mram patch: segment path is empty"))
		}
		segPath := filepath.Join(binDirpath, seg.Path)
		payload, err := os.ReadFile(segPath)
		if err != nil {
			panic(err)
		}

		var start int64
		if seg.OffsetBytes != nil {
			start = *seg.OffsetBytes
		} else {
			if seg.Symbol == "" {
				panic(errors.New("pimdl mram patch: segment needs symbol or offset_bytes"))
			}
			symVA, ok2 := addresses[seg.Symbol]
			if !ok2 {
				panic(fmt.Errorf(
					"pimdl mram patch: symbol %q not found in addresses.txt",
					seg.Symbol,
				))
			}
			start = symVA - anchorVA
		}

		if start < 0 || int64(len(lines)) < start+int64(len(payload)) {
			panic(fmt.Errorf(
				"pimdl mram patch: %s: start=%d len_payload=%d mram_lines=%d",
				seg.Path, start, len(payload), len(lines),
			))
		}

		for i := 0; i < len(payload); i++ {
			lines[start+int64(i)] = strconv.Itoa(int(payload[i]))
		}
	}

	if err := writePimdlMRAMLines(mramPath, lines); err != nil {
		panic(err)
	}
	fmt.Printf("pimdl: patched mram.bin using %s\n", pimdlMRAMManifestName)
}

func parsePimdlAddresses(path string) map[string]int64 {
	data, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	out := make(map[string]int64)
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			panic(fmt.Errorf("pimdl mram patch: bad addresses line: %q", line))
		}
		name := strings.TrimSpace(parts[0])
		addrStr := strings.TrimSpace(parts[1])
		v, err := strconv.ParseInt(addrStr, 10, 64)
		if err != nil {
			panic(err)
		}
		out[name] = v
	}
	return out
}

func readPimdlMRAMLines(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var lines []string
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		lines = append(lines, line)
	}
	return lines, nil
}

func writePimdlMRAMLines(path string, lines []string) error {
	out := strings.Builder{}
	for i, ln := range lines {
		if i > 0 {
			out.WriteByte('\n')
		}
		out.WriteString(ln)
	}
	out.WriteByte('\n')
	return os.WriteFile(path, []byte(out.String()), 0644)
}
