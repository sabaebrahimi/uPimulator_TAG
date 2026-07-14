# PIM-DL ↔ uPIMulator Co-Simulation (Phase 1)

Offline artifact-handoff co-simulation for the TAG project. PIM-DL native is the
ground-truth host that runs the full transformer ping-pong; uPIMulator replays each
PIM↔host **projection boundary** (QKV, O, FFN1, FFN2) on its cycle-accurate DPU+MRAM
and **bit-matches** its DPU output against PIM-DL's. uPIMulator's output is only
cross-checked, never fed forward, so the four boundaries are independent units.

Phase 1 delivers: the ported LUT kernel, the four projection microbenchmarks,
per-boundary data handoff, the bit-match, and host↔MRAM **data-transfer cycle
accounting** (H2D lut/index + D2H output via `SimulateMemory`). The host-side
*reorder* timing cost model (and TAG) are the next phase; the emitter already
dumps the pre-reorder DPU-tiled output so that step is a pure add.

## The downscaled STATIC config
The real QKV static LUT tile is 2.3 MB, but the whole static LUT buffer must fit the
**real 64 KB DPU WRAM** (alongside 16 tasklet stacks) so PIM-DL's native DPU binary
links — uPIMulator itself models 128 KB, but the bit-match requires PIM-DL to build the
same config. So both sides run a downscaled STATIC (`lut_load_type=0`) + NFC
(`loop_order=0`) config with every LUT ≤ 24 KB. It MUST match on both sides.
Source of truth = three places that must agree:

| Projection | benchmark      | (num_codebook, feature_stile) | LUT   | uPIMulator dpu/CMakeLists.txt | PIM-DL config |
|------------|----------------|-------------------------------|-------|-------------------------------|---------------|
| QKV        | `PIMDL`        | cb=32, fs=48                  | 24 KB | `benchmark/PIMDL/dpu`         | `cosim_static_small.yaml` qkv |
| O          | `PIMDL_O`      | cb=32, fs=16                  |  8 KB | `benchmark/PIMDL_O/dpu`       | ... o   |
| FFN1       | `PIMDL_FFN1`   | cb=32, fs=32                  | 16 KB | `benchmark/PIMDL_FFN1/dpu`    | ... ffn1|
| FFN2       | `PIMDL_FFN2`   | cb=64, fs=16                  | 16 KB | `benchmark/PIMDL_FFN2/dpu`    | ... ffn2|

Common geometry: `N_STILE=N_MTILE=64`, `NUM_CENTROID=16`, `CB_MTILE=16`, `NR_TASKLETS=16`.
The three places also include `tools/build_dpu_local.sh`'s `CFG` table (Docker-free build).

## Files

### uPIMulator (`golang_vm/uPIMulator`)
- `benchmark/PIMDL/dpu/task.c` — the ported real LUT kernel (STATIC + NFC), macro-driven.
- `benchmark/PIMDL/support/common.h` — PIM-DL data types (int8/uint16/int32).
- `benchmark/PIMDL{,_O,_FFN1,_FFN2}/` — four projection microbenchmarks (siblings `#include` the one kernel).
- `benchmark/*/host/app.c` — minimal host: alloc/load/args/launch (data comes via mram-patch).
- `src/program/pimdl_mram_patch.go` — splices LUT/index into `mram.bin`; anchor `MRAM_BASE`; reads from `--pimdl_patch_dirpath`.
- `src/host/vm/virtual_machine.go` — `DumpPimdlOutput()` writes `output_data` → `pimdl_output.bin` when `PIMDL_OUTPUT_BYTES` is set.
- `src/main.go` — new options `--pimdl_patch_dirpath`, `--skip_compile`.
- `tools/build_dpu_local.sh`, `tools/cosim_prepare.py`, `tools/cosim_check.py`, `tools/run_cosim.sh`.

### PIM-DL (`PIM-DL-ASPLOS/inference-engine`)
- `src/host/pim_lut_host.cpp` — co-sim emitter taps (guarded by `PIMDL_COSIM_DUMP`).
- `src/host/CMakeLists.txt` — `-DPIMDL_COSIM_DUMP=ON` option.
- `configs/cosim_static_small.yaml` — the downscaled STATIC config.
- `run_cosim_emit.py` — build-with-emitter + run one layer + collect snapshots.

## Running it

### 1. PIM-DL native — emit the ground-truth snapshots (your UPMEM env)
```bash
cd PIM-DL-ASPLOS/inference-engine
python3 run_cosim_emit.py --dump_dir cosim_dumps
# -> cosim_dumps/{lut,index,output}_cb{CB}_fs{FS}.bin  for each projection
#    (qkv cb32_fs48 | o cb32_fs16 | ffn1 cb32_fs32 | ffn2 cb64_fs16)
```

### 2. uPIMulator — replay + bit-match
```bash
cd golang_vm/uPIMulator
go build -o build/uPIMulator ./src

# Docker-free DPU build with a local UPMEM SDK (default 2025.1.0):
UPMEM_HOME=/home/saba/master/upmem-2025.1.0-Linux-x86_64 tools/build_dpu_local.sh

# replay all four boundaries and bit-match (reuses prebuilt sdk/build assembly)
tools/run_cosim.sh ../../PIM-DL-ASPLOS/inference-engine/cosim_dumps
```
Expected: `PASS [qkv] / PASS [o] / PASS [ffn1] / PASS [ffn2]`, plus per-projection DPU
logic/memory cycles and `HostTransfer_*` H2D/D2H transfer cycles from `bin_cosim/log.txt`.

To use the stock Docker/UPMEM compile instead of the local build, run with
`SKIP_COMPILE=0 tools/run_cosim.sh ...` (and omit `build_dpu_local.sh`).

## Notes
- uPIMulator runs one DPU per config; the bit-match is DPU 0's shard vs PIM-DL's DPU 0.
- Host↔MRAM **data-transfer cycles are charged** (Phase A): timed HOST_TO_DEVICE for
  `lut_table`/`input_index` before launch, and timed DEVICE_TO_HOST for `output_data`
  readback. Look for `HostTransfer_*` in `bin_cosim/log.txt` (`h2d_cycle`, `d2h_cycle`,
  `transfer_cycle`, `*_bytes`). The mram-patch still supplies functional correctness;
  the timed path re-drives the same bytes through uPIMulator's DRAM model.
- Host **reorder** timing (strided CPU scatter/gather) and TAG remain the next phase.
