#!/usr/bin/env python3
"""
Bit-match uPIMulator's simulated DPU-tiled output against PIM-DL native's dumped
DPU-tiled output for one projection. A PASS proves the ported kernel reproduces
PIM-DL's DPU compute exactly for that boundary.

    uPIMulator side: <bin_dir>/pimdl_output.bin        (from the Go dump)
    PIM-DL side:     <dump_dir>/output_cb{cb}_fs{fs}.bin (from the emitter)
"""
import argparse
import os
import sys

PROJECTIONS = {          # projection -> (num_codebook, feature_stile)
    "qkv":  (32, 48),
    "o":    (32, 16),
    "ffn1": (32, 32),
    "ffn2": (64, 16),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("projection", choices=sorted(PROJECTIONS.keys()))
    ap.add_argument("--bin_dir", required=True, help="uPIMulator bin dir (holds pimdl_output.bin)")
    ap.add_argument("--dump_dir", required=True, help="PIM-DL emitter dump dir")
    args = ap.parse_args()

    cb, fs = PROJECTIONS[args.projection]
    sim_path = os.path.join(args.bin_dir, "pimdl_output.bin")
    ref_path = os.path.join(args.dump_dir, f"output_{args.projection}_cb{cb}_fs{fs}.bin")
    if not os.path.exists(ref_path):
        ref_path = os.path.join(args.dump_dir, f"output_cb{cb}_fs{fs}.bin")

    for p in (sim_path, ref_path):
        if not os.path.exists(p):
            sys.exit(f"FAIL [{args.projection}]: missing {p}")

    with open(sim_path, "rb") as f:
        sim = f.read()
    with open(ref_path, "rb") as f:
        ref = f.read()

    if len(sim) != len(ref):
        sys.exit(f"FAIL [{args.projection}]: size {len(sim)} (sim) != {len(ref)} (ref)")

    if sim != ref:
        # locate first differing int32 for a useful message
        import struct
        n = len(sim) // 4
        for i in range(n):
            a = struct.unpack_from("<i", sim, 4 * i)[0]
            b = struct.unpack_from("<i", ref, 4 * i)[0]
            if a != b:
                sys.exit(f"FAIL [{args.projection}]: first mismatch at int32 index {i}: "
                         f"sim={a} ref={b}")
        sys.exit(f"FAIL [{args.projection}]: byte mismatch")

    print(f"PASS [{args.projection}]: {len(sim)} bytes bit-match "
          f"({len(sim)//4} int32 outputs)")


if __name__ == "__main__":
    main()
