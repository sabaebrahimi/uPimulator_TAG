#!/usr/bin/env python3
"""
uPIMulator TAG co-simulation: PIM-DL native emitter driver (Phase 1).

Runs one transformer layer natively (PIM-DL, ground truth) with the co-sim emitter
compiled in (PIMDL_COSIM_DUMP), producing per-projection snapshots of the exact
per-DPU bytes exchanged at each PIM<->host boundary:

    <dump_dir>/lut_cb{CB}_fs{FS}.bin      (int8  LUT shard, DPU 0)
    <dump_dir>/index_cb{CB}_fs{FS}.bin    (uint16 index shard, DPU 0, post pre-multiply)
    <dump_dir>/output_cb{CB}_fs{FS}.bin   (int32 DPU-tiled output, DPU 0, pre-reorder)

These feed uPIMulator (lut/index via the mram-patch; output is the bit-match golden
reference). Signatures (CB,FS) uniquely identify each projection in cosim_static_small.yaml:
    QKV cb=64 fs=48 | O cb=64 fs=16 | FFN1 cb=64 fs=64 | FFN2 cb=256 fs=16

Reuses run_layer.compile() so the (huge) tile-macro derivation is never duplicated,
then augments the existing CMake cache with -DPIMDL_COSIM_DUMP=ON and rebuilds.

Usage:
    python3 run_cosim_emit.py [--config configs/cosim_static_small.yaml] [--dump_dir cosim_dumps]
"""
import argparse
import os
import subprocess
import types

import run_layer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(BASE_DIR, "build")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(BASE_DIR, "configs", "cosim_static_small.yaml"))
    ap.add_argument("--dump_dir", default=os.path.join(BASE_DIR, "cosim_dumps"))
    args = ap.parse_args()

    dump_dir = os.path.abspath(args.dump_dir)
    os.makedirs(dump_dir, exist_ok=True)

    # 1) Compile the transformer (all four DPU binaries + host lib) via the stock
    #    derivation. This populates the CMake cache with every tile macro.
    compile_args = types.SimpleNamespace(
        transformer_config_file=args.config,
        need_compile=True,
        need_measure_energy=False,
        need_breakdown=False,
        breakdown_type=0,
    )
    run_layer.compile(compile_args)

    # 2) Turn on the emitter by augmenting the existing cache (no need to re-pass the
    #    macros; CMake remembers them) and rebuild just what changed.
    subprocess.run(["cmake", "-DPIMDL_COSIM_DUMP=ON", BASE_DIR], cwd=BUILD_DIR, check=True)
    subprocess.run(["make", "-j"], cwd=BUILD_DIR, check=True)

    # 3) Run one layer with the dump directory set. test_transformer_layer calls
    #    pim_lut() for QKV, O, FFN1, FFN2; the emitter writes one snapshot per boundary.
    env = dict(os.environ)
    env["PIMDL_COSIM_DUMP_DIR"] = dump_dir
    # Dump-only mode: still executes all 4 LUT kernels (QKV/O/FFN1/FFN2) and emits
    # their snapshots, but skips the full attention/norm/residual host path so cosim
    # snapshot generation is robust under downscaled configs.
    env["PIMDL_COSIM_DUMP_ONLY"] = "1"
    binary = os.path.join(BUILD_DIR, "bin", "test_transformer_layer")
    print(f"[cosim] running {binary} with PIMDL_COSIM_DUMP_DIR={dump_dir}")
    subprocess.run([binary, args.config], cwd=BASE_DIR, env=env, check=True)

    print("\n[cosim] emitted snapshots:")
    for fn in sorted(os.listdir(dump_dir)):
        if fn.endswith(".bin"):
            print(f"    {os.path.join(dump_dir, fn)}  ({os.path.getsize(os.path.join(dump_dir, fn))} bytes)")
    print("\n[cosim] next: feed these to uPIMulator with tools/run_cosim.sh")


if __name__ == "__main__":
    main()
