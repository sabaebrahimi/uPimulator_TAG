#!/usr/bin/env python3
"""
Stage PIM-DL's emitted per-DPU snapshot for one projection into a uPIMulator
mram-patch directory: copies the LUT/index shards to pimdl_segments/ and writes
pimdl_mram_patch.json (anchored at MRAM_BASE, targeting the named MRAM symbols).

Prints PIMDL_OUTPUT_BYTES=<n> so the caller can export it for the simulator run
(the Go dump writes exactly that many bytes of output_data to pimdl_output.bin).

The projection table mirrors configs/cosim_static_small.yaml and the benchmark
dpu/CMakeLists.txt tile macros; the (cb, fs) signature selects the emitter files.
"""
import argparse
import json
import os
import shutil
import sys

# projection -> (benchmark, num_codebook, feature_stile, n_stile, num_centroid)
PROJECTIONS = {
    "qkv":  ("PIMDL",       32, 48, 64, 16),
    "o":    ("PIMDL_O",     32, 16, 64, 16),
    "ffn1": ("PIMDL_FFN1",  32, 32, 64, 16),
    "ffn2": ("PIMDL_FFN2",  64, 16, 64, 16),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("projection", choices=sorted(PROJECTIONS.keys()))
    ap.add_argument("--dump_dir", required=True, help="PIM-DL emitter output dir")
    ap.add_argument("--patch_dir", required=True, help="output dir for manifest + segments")
    args = ap.parse_args()

    bench, cb, fs, n_stile, ct = PROJECTIONS[args.projection]

    lut_bytes = fs * cb * ct * 1        # int8
    index_bytes = n_stile * cb * 2      # uint16
    output_bytes = n_stile * fs * 4     # int32

    # Prefer tag-qualified dump names (emitter sets PIMDL_COSIM_DUMP_TAG per
    # projection), and fall back to legacy cb/fs-only names for compatibility.
    src_lut = os.path.join(args.dump_dir, f"lut_{args.projection}_cb{cb}_fs{fs}.bin")
    src_index = os.path.join(args.dump_dir, f"index_{args.projection}_cb{cb}_fs{fs}.bin")
    if not os.path.exists(src_lut):
        src_lut = os.path.join(args.dump_dir, f"lut_cb{cb}_fs{fs}.bin")
    if not os.path.exists(src_index):
        src_index = os.path.join(args.dump_dir, f"index_cb{cb}_fs{fs}.bin")
    for p, want in ((src_lut, lut_bytes), (src_index, index_bytes)):
        if not os.path.exists(p):
            sys.exit(f"error: missing emitter dump {p} (run run_cosim_emit.py first)")
        got = os.path.getsize(p)
        if got != want:
            sys.exit(f"error: {p} is {got} bytes, expected {want} "
                     f"(config mismatch between PIM-DL emitter and uPIMulator)")

    seg_dir = os.path.join(args.patch_dir, "pimdl_segments")
    os.makedirs(seg_dir, exist_ok=True)
    shutil.copyfile(src_lut, os.path.join(seg_dir, "lut_table.bin"))
    shutil.copyfile(src_index, os.path.join(seg_dir, "input_index.bin"))

    manifest = {
        "anchor": "MRAM_BASE",
        "segments": [
            {"symbol": "lut_table",   "path": "pimdl_segments/lut_table.bin"},
            {"symbol": "input_index", "path": "pimdl_segments/input_index.bin"},
        ],
    }
    with open(os.path.join(args.patch_dir, "pimdl_mram_patch.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[prepare] {args.projection} ({bench}): lut={lut_bytes}B index={index_bytes}B "
          f"patch_dir={args.patch_dir}", file=sys.stderr)
    # stdout is machine-readable for the orchestrator.
    print(f"BENCHMARK={bench}")
    print(f"PIMDL_OUTPUT_BYTES={output_bytes}")


if __name__ == "__main__":
    main()
