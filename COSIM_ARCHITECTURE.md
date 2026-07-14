# PIM-DL ↔ uPIMulator Co-Simulation — Architecture

This document explains *why* the co-simulation is shaped the way it is, the data flow
end-to-end, and the design decisions behind each piece. For the command-line recipe see
`COSIM_README.md`; this file is the "why".

---

## 1. The problem we are attacking

PIM-DL runs a transformer as a **ping-pong between the DPUs (PIM) and the host CPU**:

```
        ┌──────────── one transformer layer ────────────────────────────────┐
        │                                                                    │
  QKV proj (PIM) ─▶ readback + REORDER (host) ─▶ attention/ggml (host) ─▶ index_calc
        ▲                                                                    │
        │                                                                    ▼
    (weights)                                    O proj (PIM) ─▶ readback + REORDER ─▶
                                                 FFN1 (PIM) ─▶ REORDER ─▶ gelu ─▶
                                                 FFN2 (PIM) ─▶ REORDER ─▶ layer out
        └────────────────────────────────────────────────────────────────────┘
```

After each PIM projection, each DPU holds its result in a **DPU-tiled, DPU-local layout**.
The host reads it back and **reorders** it into the contiguous logical layout
(`[head_dim, seq_len, head_num, batch]`) that ggml attention needs. That reorder is a
strided, poor-locality `memcpy` through DRAM and — per the project's breakdown — can cost
as much as or more than the attention itself. The same "DPU-tiled → logical" transform
recurs at **five sites** in a layer (QKV→Q/K/V incl. the V transpose, O, O→FFN, FFN1
activation, FFN2→output).

**TAG** (the eventual hardware) folds that reorder into the host↔MRAM transfer. The
`q/k/v_offset` address formulas in `transformer_layer.cpp` *are* TAG's address-generation
spec. The goal of the larger project is to remove the host-side reorder overhead.

**Phase 1 (this delivery)** builds the vehicle to study it: get the real LUT kernel running
cycle-accurately in uPIMulator at each projection boundary, fed by PIM-DL's real data, and
prove functional equivalence. Reorder *timing* and TAG come next.

---

## 2. Why a co-simulation (and not one simulator)

Two hard facts force the split:

- **uPIMulator cannot run the host math.** Its "host" is a restricted bytecode VM that only
  orchestrates `dpu_*` calls. It has no ggml, no floating-point host compute, no reorder.
  It *does* model, cycle-accurately: (a) DPU execution and (b) the host↔MRAM/DRAM transfer path.
- **PIM-DL's host is real C++** (ggml, cnpy, float reorder). It cannot be ported into the VM.

So each simulator keeps what it is good at:

| Domain                         | Runs in            | Fidelity                    |
|--------------------------------|--------------------|-----------------------------|
| DPU kernel + MRAM (the "PIM")  | **uPIMulator**     | cycle-accurate              |
| Host reorder + ggml attention  | **PIM-DL native**  | real C++ (ground truth)     |

The coupling model is **offline artifact handoff** (not a live per-call bridge): PIM-DL
runs the whole layer and dumps a snapshot at each boundary; uPIMulator replays each boundary
from its snapshot. This matches the existing scaffolding (mram-patch, per-benchmark dirs) and
needs no refactor of uPIMulator into a steppable server.

---

## 3. The key insight: boundaries decompose into independent units

The ping-pong looks like it forces a round-trip (O's input depends on attention, which
depends on the QKV reorder…). It does **not**, because of two properties:

1. **PIM-DL native owns all cross-stage dataflow.** It runs the true, correct chain and has
   the correct *input* for every boundary. uPIMulator is *fed* each boundary's input from
   PIM-DL's dump; it never has to produce data for the next stage.
2. **uPIMulator's output is only cross-checked, never fed forward.** TAG is required to
   produce *bit-identical* reordered output (the functional anchor), so everything downstream
   is unaffected by construction. Correctness therefore reduces to a **per-boundary bit-match**.

Consequences:

- Each of the four projections (QKV, O, FFN1, FFN2) is an **independent, measurable unit**.
- The ggml attention that sits *between* PIM stages is host compute uPIMulator can't run —
  and doesn't need to. It is outside TAG's scope, identical in baseline and TAG, so it's
  taken as a constant from PIM-DL native and never simulated.
- Totals compose: `DPU cycles = Σ per-boundary`, and (next phase) `reorder overhead = Σ per-boundary`.

```
   PIM-DL native (full layer, ground truth)
        │  at each boundary, dump DPU-0 shard:
        │    lut_cb{CB}_fs{FS}.bin      (LUT the DPUs saw)
        │    index_cb{CB}_fs{FS}.bin    (indices the DPUs saw, post pre-multiply)
        │    output_cb{CB}_fs{FS}.bin   (DPU-tiled output, PRE-reorder = golden ref)
        ▼
   ┌─────────────── per boundary, independently ───────────────┐
   │ uPIMulator: run the SAME kernel on cycle-accurate DPU+MRAM │
   │   inputs (lut,index) spliced into MRAM via the mram-patch  │
   │   output_data read back from simulated MRAM                │
   │   bit-match output_data == PIM-DL's output_cb/fs  ✅/❌     │
   └────────────────────────────────────────────────────────────┘
```

---

## 4. End-to-end data flow

```
PIM-DL-ASPLOS/inference-engine
  run_cosim_emit.py
     ├─ run_layer.compile()               # builds 4 DPU binaries + host lib
     ├─ cmake -DPIMDL_COSIM_DUMP=ON; make  # turn on emitter taps
     └─ test_transformer_layer cosim_static_small.yaml   (PIMDL_COSIM_DUMP_DIR=cosim_dumps)
             │
             │  pim_lut_host.cpp taps (guarded by PIMDL_COSIM_DUMP):
             │    after prepare_lut_table()   → lut_cb{CB}_fs{FS}.bin
             │    after prepare_input_index() → index_cb{CB}_fs{FS}.bin
             │    after FROM_DPU readback     → output_cb{CB}_fs{FS}.bin  (pre-reorder)
             ▼
        cosim_dumps/*.bin   ← the handoff artifacts

golang_vm/uPIMulator  (tools/run_cosim.sh, per projection)
  cosim_prepare.py  ── copies lut/index shard → <patch_dir>/pimdl_segments/,
                        writes pimdl_mram_patch.json (anchor MRAM_BASE),
                        emits BENCHMARK + PIMDL_OUTPUT_BYTES
             ▼
  uPIMulator --benchmark PIMDL[_O/_FFN1/_FFN2] --pimdl_patch_dirpath <patch_dir>
             │   compile/link the ported kernel → mram.bin, addresses.txt
             │   RunPimdlMramPatch: splice lut_table / input_index into mram.bin
             │   DpuLaunch: cycle-accurate kernel execution (LUT accumulation)
             │   DumpPimdlOutput: read output_data from MRAM → bin/pimdl_output.bin
             ▼
  cosim_check.py  ── bit-match pimdl_output.bin == cosim_dumps/output_cb{CB}_fs{FS}.bin
```

The two sides meet at exactly two byte-streams per boundary: the **inputs** (LUT+index,
handed PIM-DL→uPIMulator) and the **output** (handed uPIMulator→checker, compared to PIM-DL).

---

## 5. Component reference

### uPIMulator (`golang_vm/uPIMulator`)
- **`benchmark/PIMDL/dpu/task.c`** — the ported real LUT kernel (`STATIC_LUT_TABLE` +
  `LOOP_ORDER_NFC`). The `lut_kernel()` body is byte-for-byte identical to PIM-DL's so the
  compute matches exactly. Tile sizes are `-D` macros, exactly like PIM-DL.
- **`benchmark/PIMDL{,_O,_FFN1,_FFN2}`** — four projection microbenchmarks. Each is one
  `--benchmark` run (uPIMulator maps a benchmark name 1:1 to a DPU binary). Siblings
  `#include "../../PIMDL/dpu/task.c"` so there is a single kernel source; only the tile
  macros in each `dpu/CMakeLists.txt` differ.
- **`benchmark/*/host/app.c`** — minimal host DSL: alloc → load → push `dpu_arguments` →
  launch. It does **not** push LUT/index (those arrive via the mram-patch) and does not read
  `output_data` in the DSL (see §6).
- **`src/program/pimdl_mram_patch.go`** — splices external segment files into the linked
  `mram.bin`. Reads manifest + segments from `--pimdl_patch_dirpath` (see §6 on why it's
  separate). Anchor `MRAM_BASE` resolves to the MRAM image base.
- **`src/host/vm/virtual_machine.go` → `DumpPimdlOutput()`** — after simulation, reads
  `output_data` from simulated MRAM and writes `pimdl_output.bin`. Active only when
  `PIMDL_OUTPUT_BYTES` is set.
- **`src/main.go`** — `--pimdl_patch_dirpath`, `--skip_compile`.
- **`tools/`** — `build_dpu_local.sh` (Docker-free DPU build), `cosim_prepare.py`,
  `cosim_check.py`, `run_cosim.sh`.

### PIM-DL (`PIM-DL-ASPLOS/inference-engine`)
- **`src/host/pim_lut_host.cpp`** — emitter taps, guarded by `PIMDL_COSIM_DUMP`. Dumps DPU 0's
  shard at the three points above. Files are tagged by `(num_codebook, feature_stile_size)`,
  which is unique per projection, so dumps are order-independent.
- **`src/host/CMakeLists.txt`** — `-DPIMDL_COSIM_DUMP=ON` option.
- **`configs/cosim_static_small.yaml`** — the downscaled STATIC config (see §7).
- **`run_cosim_emit.py`** — build-with-emitter and run one layer.

---

## 6. Two mechanics that shape the design

**Named MRAM symbols vs the VM transfer model.** The ported kernel uses named
`__mram_noinit` arrays (`lut_table`, `input_index`, `output_data`), faithful to PIM-DL. But
the VM's DSL `dpu_push_xfer` routes **string** symbol names through the *WRAM* path; only the
MRAM **heap** is addressable by a numeric base. So the host DSL cannot push/read these named
MRAM symbols. Therefore:
- **Inputs** are injected by splicing `mram.bin` directly (the mram-patch), which operates on
  the linked image by symbol address and sidesteps the transfer model entirely.
- **Output** is extracted Go-side (`DumpPimdlOutput`) by reading simulated MRAM at
  `addresses["output_data"]`, again bypassing the DSL.

**Why a separate patch directory.** `main.go` wipes `bin_dirpath` at startup (`os.RemoveAll`),
then compile/link repopulate it, then the patch hook runs. Anything pre-placed in
`bin_dirpath` would be deleted before the hook sees it. So the manifest + segments live in a
stable `--pimdl_patch_dirpath`, while `mram.bin`/`addresses.txt` stay in `bin_dirpath`.

**MRAM addressing / the `MRAM_BASE` anchor.** The linker emits symbol VAs; `mram.bin[0]`
corresponds to VA `MramOffset` (512 KB), and `mram.bin` spans up to `__sys_used_mram_end`. A
segment's byte offset into `mram.bin` is `symbolVA − MramOffset`. The patch's `MRAM_BASE`
anchor resolves to `MramOffset`, which is always available (unlike a linked symbol name that
may be absent from `addresses.txt`). The same VA is used directly as the MRAM read address
when dumping `output_data`.

---

## 7. The downscaled STATIC config (a hard constraint)

`STATIC_LUT_TABLE` loads the **whole** LUT tile into WRAM. At real BERT-base QKV sizes that
tile is `feature_stile·num_codebook·num_centroid = 288·512·16 ≈ 2.3 MB`. The binding limit is
the **real 64 KB DPU WRAM**: even though uPIMulator models 128 KB, the bit-match requires
PIM-DL to build the *same* config, and PIM-DL's native DPU binary must link the LUT buffer
plus 16 tasklet stacks into 64 KB. Empirically the static LUT buffer must stay ≲ 32 KB; we
keep every LUT ≤ 24 KB. So both sides run a **downscaled-but-representative** STATIC config,
keeping `NUM_CENTROID=16` and the `n_stile=n_mtile=64` geometry real and shrinking
`num_codebook`/`feature_stile`.

This config is the **single source of truth in three places that must agree**, or the
bit-match is meaningless:

| Projection | benchmark    | (num_codebook, feature_stile) | LUT   |
|------------|--------------|-------------------------------|-------|
| QKV        | `PIMDL`      | cb=32, fs=48                  | 24 KB |
| O          | `PIMDL_O`    | cb=32, fs=16                  |  8 KB |
| FFN1       | `PIMDL_FFN1` | cb=32, fs=32                  | 16 KB |
| FFN2       | `PIMDL_FFN2` | cb=64, fs=16                  | 16 KB |

The three places: uPIMulator `benchmark/<NAME>/dpu/CMakeLists.txt` macros,
`configs/cosim_static_small.yaml`, and `tools/build_dpu_local.sh`'s `CFG` table.
Derived from `token_dim=128, ffn_hidden=256, lut_parallelism=8` (see the yaml header).

Shard byte sizes (identical on both sides, so the dumps drop straight in):
`lut = fs·cb·ct` (int8), `index = n_stile·cb·2` (uint16), `output = n_stile·fs·4` (int32).

---

## 8. What Phase 1 measures — and what it defers

- **Measured, cycle-accurate:** the DPU kernel execution (LUT accumulation + its internal
  WRAM↔MRAM DMA), from `bin/log.txt` (`Logic_logic_cycle`, `MemoryController_memory_cycle`, …).
- **Measured, functional:** bit-exact `output_data` vs PIM-DL (proves the ported kernel
  reproduces PIM-DL's DPU compute).
- **Deferred to the next phase:** the host↔MRAM **reorder timing**. uPIMulator's host↔MRAM
  copy is functional-only today (no cycles charged); a host-DRAM cost model over the strided
  `q/k/v_offset` segments — and then the TAG module that folds the reorder into the transfer —
  is the next step. The emitter already dumps the *pre-reorder* DPU-tiled output, so that step
  is purely additive.

---

## 9. Verification status (component-level, in-repo)

Proven without Docker/UPMEM: `go build`/`vet`/`gofmt` clean; all four DPU kernels compile
with the local SDK; Go unit tests for the mram-patch (offsets, symbol resolution, patch-dir
separation, no-op); emitter helper + taps compile; Python prepare/check pass on synthetic data
(size validation + PASS/FAIL). The full four-way bit-match runs in an environment with a
writable `benchmark/build` (see `COSIM_README.md`).
