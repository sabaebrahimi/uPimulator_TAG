# Phase 1 Implementation Plan — Port LUT Kernel & Build QKV/O/FFN Microbenchmarks (Offline Co-Simulation)

## Goal (this iteration)
Stand up the **offline artifact-handoff co-simulation** between PIM-DL (native, ground truth) and uPIMulator
(cycle-accurate DPU + MRAM). For **all four projection boundaries** (QKV, O, FFN1, FFN2), port the *real* LUT kernel
into uPIMulator, drive it with a readback host, feed it PIM-DL's actual per-boundary data, and **bit-match**
uPIMulator's DPU output against PIM-DL's dumped DPU-tiled output.

**In scope:** kernel port, readback host, data injection, native emitter, bit-match cross-check, DPU + readback-transfer cycles.
**Out of scope (next phase):** host reorder timing cost model; TAG hardware module.

## How the PIM↔host ping-pong is handled
uPIMulator never runs the pipeline. PIM-DL native runs the *true* full chain
(QKV→reorder→attention→O→reorder→FFN1→…) and, **at each PIM↔host boundary**, dumps a snapshot
(the exact LUT + `input_index` bytes it pushed to the DPUs, plus the DPU-tiled `output_data` it read back).
uPIMulator independently replays each boundary from its snapshot. Because each boundary's *input* comes from
PIM-DL's dump and uPIMulator's output is only **cross-checked, never fed forward**, the recurrence decomposes into
four independent, measurable units. ggml attention stays entirely on the PIM-DL side (a shared constant, outside TAG's scope).

## The one hard constraint: STATIC LUT must fit 128 KB WRAM
The real QKV static tile = `FEATURE_STILE_SIZE*NUM_CODEBOOK*NUM_CENTROID` = 288·512·16 = **2.3 MB ≫ 128 KB WRAM**.
So both sides run a **downscaled-but-representative STATIC config** (fewer codebooks / smaller feature tile, `NUM_CENTROID=16` kept real).
The config must be **byte-identical on both sides** or the bit-match is meaningless.

WRAM budget (STATIC, shared buffers): `lut = FEATURE_STILE_SIZE·NUM_CODEBOOK·NUM_CENTROID·1B` (dominates) +
`input = N_MTILE_SIZE·CB_MTILE_SIZE·2B` + `output = N_MTILE_SIZE·FEATURE_MTILE_SIZE·4B`, keep ≤ ~110 KB.
Worked fitting example (tunable), per projection: `NUM_CENTROID=16`, `NUM_CODEBOOK=32`, `FEATURE_STILE_SIZE=32`,
`N_STILE_SIZE=64`, `N_MTILE_SIZE=16`, `CB_MTILE_SIZE=32`, `FEATURE_MTILE_SIZE=32`
→ LUT 16 KB + in 1 KB + out 2 KB ≈ 19 KB. (O/FFN1/FFN2 differ only in `FEATURE_STILE_SIZE`/`NUM_CODEBOOK`.)

---

## Work items

### A. Port the real LUT kernel into uPIMulator (shared, macro-driven)
- Replace placeholder `golang_vm/uPIMulator/benchmark/PIMDL/dpu/task.c` with the real kernel body from
  `PIM-DL-ASPLOS/.../src/dpu/pim_lut_kernel.c` — **only** the `STATIC_LUT_TABLE` + `LOOP_ORDER_NFC` variant
  (kernel lines 97–204) plus the required `#define`s (lines 15–41). Keep it macro-driven (`N_STILE_SIZE`, `FEATURE_STILE_SIZE`,
  `N_MTILE_SIZE`, `FEATURE_MTILE_SIZE`, `CB_MTILE_SIZE`, `NUM_CODEBOOK`, `NUM_CENTROID`) exactly like PIM-DL, so one source serves all four configs.
- Update `benchmark/PIMDL/support/common.h` to carry PIM-DL's data types (`lut_data_type=int8`, `index_data_type=uint16`,
  `output_data_type=int32`, `INDEX_SIZE/LUT_SIZE/OUTPUT_SIZE`, `dpu_arguments_t`) — mirroring `src/dpu/dpu_configs.h`.
- MRAM symbols stay `lut_table` / `input_index` / `output_data` (patch + readback targets).

### B. Four projection benchmark dirs (one build → four `task.c.o`)
- `benchmark/PIMDL` = QKV (default/base). Add siblings `benchmark/PIMDL_O`, `benchmark/PIMDL_FFN1`, `benchmark/PIMDL_FFN2`.
- Each sibling: `dpu/task.c` = `#include "../../PIMDL/dpu/task.c"` (so the object is still `task.c.o`, which the linker globs);
  `dpu/CMakeLists.txt` mirrors `benchmark/VA/dpu/CMakeLists.txt` but adds this projection's tile-size `-D` flags;
  `host/app.c` = the readback host (item C); `support/common.h`.
- Register all four in `benchmark/CMakeLists.txt` (`add_subdirectory`). Verify each emits
  `benchmark/build/<NAME>/dpu/CMakeFiles/<NAME>_device.dir/task.c.o` (what `linker.go:47-54` expects).

### C. Readback host (uPIMulator host DSL, per projection)
Model on `benchmark/VA/host/app.c` using only VM-supported builtins (`dpu_alloc/load/prepare_xfer/push_xfer/launch/free`,
`DPU_FOREACH`, `malloc`, loops). Sequence: alloc `dpu_num` DPUs → load → push `dpu_arguments` TO_DPU → `dpu_launch` →
**readback `output_data` via `dpu_push_xfer(..., DPU_XFER_FROM_DPU, "output_data", 0, size, ...)`**. Input LUT/index are *not*
pushed from the DSL — they arrive via the mram-patch (item D). The `FROM_DPU` readback is the transfer we measure.

### D. Data injection via existing mram-patch hook
- Per projection, emit `pimdl_mram_patch.json` (+ `pimdl_segments/lut_table.bin`, `input_index.bin`) into `bin_dirpath`,
  consumed by `program.RunPimdlMramPatchIfPresent` (`src/program/pimdl_mram_patch.go`, already called from `main.go:66`).
  Uses `symbol` anchoring → resolves `lut_table`/`input_index` from `addresses.txt`.
- Segment bytes come straight from the PIM-DL emitter (item E) — raw MRAM-order, post-`prepare_*` bytes.

### E. PIM-DL native emitter (the real handoff)
- Add a guarded dump (`#ifdef PIMDL_COSIM_DUMP`, output dir via env var) in
  `PIM-DL-ASPLOS/.../src/host/pim_lut_host.cpp`, tapping buffers **right before `dpu_push_xfer TO_DPU`**
  (`lut_table` after `prepare_lut_table`, `input_index` after `prepare_input_index` — capturing the STATIC index
  pre-multiply at line 59) and the **DPU-tiled `output_data` right after the `FROM_DPU` readback** (line 181), *before* reorder.
  Tag each dump with a boundary id (qkv/o/ffn1/ffn2). This guarantees uPIMulator's kernel sees byte-identical input to PIM-DL's DPUs.
- Add a STATIC downscaled config (new `configs/cosim_static_small.yaml`, `lut_load_type=0`, the fitting numbers above) and a
  driver script `run_cosim_emit.py` that runs one layer with `PIMDL_COSIM_DUMP` and lays the four snapshots into per-projection dirs.
- **You run this** (your env has the UPMEM/Docker + ggml build per `e2e_results`); I write the hook, config, and script.

### F. Bit-match extraction on the uPIMulator side (minimal, contained Go)
- After simulation, dump the simulated `output_data` MRAM region to `bin_dirpath/pimdl_output.bin` (gated to PIMDL benchmarks).
  Small addition near `VirtualMachine.Dump()` (`virtual_machine.go:3874`), reading the region at `addresses["output_data"]`
  from the simulated DPU's memory — reusing the existing `TransferFromMram`/read path.
- A checker script (`tools/cosim_check.py`) diffs `pimdl_output.bin` vs PIM-DL's dumped DPU-tiled `output_data` → **bit-match proves
  the ported kernel reproduces PIM-DL's DPU compute** for that boundary.

### G. Glue + docs
- `tools/run_cosim.sh`: for each projection, place patch+segments in `bin_dirpath`, invoke the simulator with that `--benchmark`
  and its `--num_dpus_*`/`--num_tasklets`, then run the checker; collect DPU cycles + readback stats from `log.txt`.
- Short `COSIM_README.md`: exact commands, the shared-config table, and the four-boundary story.

---

## Verification (what I can prove here vs what you run)
- **Here (no Docker/UPMEM):** `go build ./...` in `golang_vm/uPIMulator` must pass with the Go dump addition;
  statically check the ported kernel's macro arithmetic and WRAM-fit for each config; validate emitter/patch/checker logic and layouts against the code I've read.
- **Your env:** build benchmarks (Docker/UPMEM), run PIM-DL emitter, run the four uPIMulator sims, run the checker.
  Expected: four **bit-match PASS** + per-projection DPU-cycle and readback-transfer numbers in `log.txt`.

## Key risks / decisions folded in
- **Shared downscaled STATIC config** is the single source of truth for both sides (item E config == item B `-D` flags). Called out explicitly.
- **Sibling-dir + include** keeps one kernel source but satisfies the linker's `task.c.o` expectation and the 1-benchmark-per-run model.
- Emitter taps **post-`prepare_*`** bytes so STATIC index pre-multiply is captured — required for a meaningful bit-match.
- Reorder-cost modeling and TAG are deliberately deferred (your Q3 choice); the artifact schema is designed so the DPU-tiled output
  and the reorder reference are already dumped, making the next phase a pure add.
