# TAG: Transfer-time Address Generation — Problem, Solution, Implementation Plan

Status: design plan (Phase 2+). Builds directly on the Phase-1 co-simulation
(`COSIM_ARCHITECTURE.md`). This document defines the problem quantitatively, describes
TAG as a hardware-software co-design, and gives a concrete implementation plan in
uPIMulator, anchored to real files and timing primitives that already exist.

---

## 1. Problem definition

### 1.1 The original fused-stage breakdown

The original PIM-DL breakdown of **host-side (non-AMM)** time in one transformer layer was:

| Host-side component                        | % of non-AMM |
|--------------------------------------------|--------------|
| gelu + FFN1 reorder                        | **60.18%**   |
| attention CPU compute (ggml)               | 28.10%       |
| post-FFN2 reorder + norm + residual        | 4.73%        |
| post-O reorder + norm + residual           | 3.97%        |
| attention reorder                          | 2.77%        |
| tensor allocation overhead                 | 0.07%        |
| other                                      | 0.18%        |

This table groups entire stages that contain a reorder with the reorder itself. In particular,
the 60.18% entry includes GELU, while the post-O and post-FFN2 entries include normalization
and residual computation. Therefore, their **≈71.6%** sum is an upper bound for
reorder-related stages, not measured pure reordering, and TAG cannot remove the included
compute.

Follow-up measurements time only the four layout loops. Full MHA host replay measures
0.2225 ms at 1 DPU and 0.2245 ms at 8 DPUs, about 1.1% of reported host time. Medium MHA
measures 0.0145 and 0.0155 ms in controlled host replay. The corresponding 1-DPU and 8-DPU
layouts have identical payload and microtile counts, which explains the flat result. See
`golang_vm/uPIMulator/cosim_results/mha_medium/reorder_analysis.md` for the measurements and
operation-count analysis. TAG evaluation must use the pure reorder timer and transfer-pattern
cost rather than claim the fused-stage sum as removable overhead.

### 1.2 Where the overhead comes from
Each PIM projection leaves its result in **DPU-tiled, DPU-local layout** in MRAM. The
logical tensor a token `(batch, head, seq, dim)` needs is scattered across DPUs and across
tiles within each DPU. Today PIM-DL:
1. **reads back** every DPU's output in parallel (`dpu_push_xfer FROM_DPU`,
   `pim_lut_host.cpp:172-181`) into a DPU-major buffer, then
2. **reorders** it on the CPU with strided `memcpy`s into the contiguous logical layout
   (`transformer_layer.cpp:100-211` for QKV; three more sites for O, FFN1, FFN2).

The reorder is software address generation: for every element it computes "where in the
logical tensor does this DPU-tiled element belong" via the `q/k/v_offset` formulas
(`transformer_layer.cpp:114-122, 148-156, 182-190`). It contains strided host-memory accesses,
but the isolated measurements show that it is a small part of current end-to-end execution.

### 1.3 What UPMEM offers today (the two existing read methods)
TAG is framed against UPMEM's existing DPU→host read methods:
- **Method 1 — single-DPU read:** read one DPU's MRAM region contiguously.
- **Method 2 — parallel read (what PIM-DL uses):** `dpu_push_xfer` broadcasts one MRAM
  offset+size and reads the *same* region from *many* DPUs at once into a host buffer. Fast,
  but it can only produce a **DPU-major, tile-order** buffer. The logical reordering must
  then happen on the CPU. Both methods move bytes in their stored order; neither can gather.

### 1.4 The precise problem statement
> Reading DPU MRAM in its stored (DPU-tiled) order forces a subsequent CPU scatter/gather to
> reach logical layout. A **third read method** could produce the **logically-ordered** stream
> directly during transfer, but its benefit must be compared with the measured pure CPU reorder
> cost and any extra transfer cost caused by strided MRAM access.

---

## 2. Solution: TAG as a third read method (HW/SW co-design)

**TAG = Transfer-time Address Generation.** A hardware module added to the modeled UPMEM
memory path that, on a DPU→host read, consumes a **descriptor** describing the logical→physical
layout and **emits the already-reordered stream** — reading MRAM with the right strides so the
host receives data in logical order. It is UPMEM's third read method, alongside single-DPU and
parallel read.

### 2.1 Hardware side (what TAG *is*)
A descriptor-driven gather engine sitting in the DPU↔host read path. Instead of "read
[offset,size) from each DPU in tile order," it executes a **strided gather program**:

```
for each logical output segment s (contiguous run of feature_mtile_size floats):
    (dpu, mram_addr) = TAG_addr_gen(descriptor, s)   # the q/k/v_offset math, in hardware
    stream <- MRAM[dpu][mram_addr : mram_addr + seg_bytes]
```

`TAG_addr_gen` is exactly the inverse of the PIM-DL reorder loop: given a logical coordinate,
produce the DPU id and MRAM address of the tile holding it. The formulas are already written
down in `transformer_layer.cpp` — TAG implements them in the transfer path rather than in a CPU
loop after the fact.

### 2.2 Software side (what the host does instead)
The host builds a **TAG descriptor** once per boundary (from `AttentionParams`/`LUTParams` —
`seq_len, head_dim, head_num, batch, n_tile_size, token_tile_size, feature_mtile_size,
lut_parallelism, dpu_num`) and issues **one TAG read** in place of (parallel read + CPU reorder).
No per-element CPU work; the reorder cost becomes a property of the transfer.

### 2.3 Why this is the right abstraction for uPIMulator
uPIMulator does not model host CPU compute, but it *does* model the host↔MRAM transfer/DRAM
timing. A memory-path gather belongs precisely in that layer. So TAG is expressed as a
**transfer-timing cost model over an access pattern** (number of strided segments, segment size,
stride distances, row-buffer locality), not as CPU instructions. This is the substrate uPIMulator
is built to measure.

### 2.4 The two modes we compare
- **`tag=off` (baseline):** parallel read (contiguous, cheap transfer) **+** a modeled CPU
  reorder cost = the strided access pattern charged against the DRAM timing model. This is the
  faithful model of PIM-DL today.
- **`tag=on`:** one descriptor-driven strided transfer; the reorder is folded in, the separate
  CPU-reorder cost is removed. TAG's own timing (address-gen throughput, setup latency) is charged.

The headline result is `cycles(tag=off) − cycles(tag=on)` at each of the 4 reorder sites, and
the functional anchor is that `tag=on`'s reordered output **bit-matches** the CPU reorder.

---

## 3. What already exists in uPIMulator that we build on

The plan reuses real, present machinery (verified in tree):

- **Device DPU MRAM timing** (`src/device/simulator/dpu/dram/`): `RowBuffer` charges
  `t_ras/t_rcd/t_cl/t_bl/t_rp` per activation/read/precharge and counts
  `num_activations/num_reads/num_precharges/read_bytes` — the exact row-buffer-locality model a
  strided gather needs. `MemoryController.Read(addr,size)` already walks wordlines.
- **Host↔MRAM transfer model** (`src/host/vm/dram/`): `TransferCommand`, `MemoryController`,
  `MemoryScheduler`, and the conventional-DRAM `channel`/`rank`/`bank` models. Crucially,
  `VirtualMachine.SimulateMemory()` + `Checkpoint()` (`virtual_machine.go:3792,3821`) already
  implement a cycle-driven host transfer loop over `push_xfer` — **but are currently dead code
  (never called).** TAG turns this dormant path on and drives it with a TAG-aware access pattern.
- **Co-sim data path** (Phase 1): the mram-patch injects real per-DPU LUT/index; the ported
  kernel produces real DPU-tiled `output_data`; `DumpPimdlOutput` extracts it. TAG reads *that*
  simulated MRAM, so the gathered stream is real data we can bit-match.
- **Config plumbing**: `main.go` command-line options + `StatFactory` for reporting.

We are not inventing a timing model from scratch — we are adding an access-pattern generator and
switching on an existing transfer-timing loop.

---

## 4. Implementation plan

Five phases. Each is independently testable; Go builds are verified locally, full runs use the
Phase-1 co-sim vehicle.

### Phase A — Turn on host transfer timing for the readback (baseline plumbing) ✅
**Goal:** make the DPU→host readback actually cost cycles, so "reorder overhead" has a number.
- Wire `DpuTransfer` (`virtual_machine.go`) DEVICE_TO_HOST / HOST_TO_DEVICE MRAM path to
  enqueue a `TransferCommand` into `push_xfer` and drive `SimulateMemory()` (was dead),
  instead of the functional-only `TransferFromMram` / `TransferToMram`.
- Co-sim: timed H2D for `lut_table`/`input_index` before launch + timed D2H in
  `DumpPimdlOutput` (`pimdl_transfer.go`).
- Add a `HostTransfer` stat group (cycles, bytes) via `StatFactory`.
- **Deliverable:** `log.txt` shows nonzero, byte-proportional `HostTransfer_*` cycles.
- **Test:** `o` projection: H2D 12288 B → 29220 cycles, D2H 4096 B → 11677 cycles; bit-match PASS.

### Phase B — Model the baseline CPU reorder as a transfer/DRAM cost
**Goal:** represent the exact reorder access pattern and calibrate its host cost against the
isolated native timers.
- New package `src/host/vm/tag/` with a **descriptor** type capturing one reorder site:
  `AccessPattern{ segments []Segment }`, `Segment{ dpu int; mramAddr, sizeBytes int64;
  logicalAddr int64 }`. A builder `BuildReorderPattern(params, site)` enumerates segments by
  porting the `q/k/v_offset` loops (`transformer_layer.cpp`) — one segment per
  `feature_mtile_size` run.
- A cost model `ReorderCost(pattern)` calibrated to the native CPU reorder measurements. CPU
  reordering accesses host memory, so DPU MRAM row-buffer timing must not be used as its baseline
  cost. The DPU DRAM model is used separately for TAG's gather reads.
- Expose site parameters to the host: pass `AttentionParams` fields via `dpu_arguments` /
  a sidecar JSON in the patch dir (co-sim already has a patch dir), so uPIMulator knows the
  geometry without running ggml.
- **Deliverable:** per-site `reorder_baseline_cycles` in `log.txt`; their ratios and total must
  track the four pure reorder timers, excluding GELU, normalization, and residual compute.
- **Test:** a Go unit test on `BuildReorderPattern` — the enumerated `(logicalAddr → dpu,mramAddr)`
  map must be a bijection identical to the C++ reorder for a small config (compare against a
  reference table generated from the PIM-DL formulas).

### Phase C — Implement the TAG hardware module
**Goal:** the third read method.
- In `src/host/vm/tag/`, add `TagEngine` with two pluggable pieces the spec will refine:
  1. **Descriptor/addressing model** — `TagDescriptor` (the compact form of the same layout
     mapping: base geometry + strides, not an explicit segment list) and `TagAddrGen(desc, i) →
     (dpu, mramAddr)` producing the i-th logical segment's source. This is `TAG_addr_gen` from §2.1.
  2. **Timing model** — `TagTransferCost(desc)`: descriptor-driven streamed read. Parameters:
     address-gen throughput (segments/cycle), setup/latency to load a descriptor, and how many
     DPUs it can gather from in parallel (models that TAG reads across DPUs like the parallel-read
     method, but with per-DPU strided addressing). Charged against the same `RowBuffer`/channel
     timing so it's comparable to Phase A/B.
- A `tag` mode switch (`--tag off|on`, `main.go`). `on` routes the readback through `TagEngine`
  (gather + TAG timing) and **skips** the Phase-B reorder cost; `off` = parallel read + Phase-B
  reorder cost.
- **Deliverable:** for each site, `tag_on_transfer_cycles` vs `tag_off_total_cycles` in `log.txt`.
- **Test:** `TagAddrGen` and `BuildReorderPattern` must induce the **same** logical→physical
  mapping (Go test asserting equality) — TAG reproduces the exact reorder addresses.

### Phase D — Functional correctness (bit-match the gather)
**Goal:** prove TAG's reordered output is byte-identical to the CPU reorder.
- `TagEngine.Gather()` reads the simulated MRAM (via the device `MemoryController.Read`, as
  `DumpPimdlOutput` does) following `TagAddrGen`, producing a logically-ordered buffer.
- Extend the Phase-1 checker: PIM-DL native already dumps the DPU-tiled output; add a native dump
  of the **post-reorder** logical tensor (a second tap in `transformer_layer.cpp` after the
  reorder loop, guarded by `PIMDL_COSIM_DUMP`). `cosim_check.py` gains a `--reordered` mode that
  compares TAG's gathered buffer against PIM-DL's post-reorder reference.
- **Deliverable:** `PASS [<site>] reorder bit-match` for all 4 sites.
- **Test:** the existing PASS/FAIL harness, extended; a deliberate stride error must FAIL.

### Phase E — Evaluation sweeps
**Goal:** the paper numbers.
- Sweep `seq_len, head_dim, batch, n_tile_size, lut_parallelism` and report, per site and summed:
  baseline (parallel read + reorder) vs TAG total cycles, and the reorder/transfer component.
- Show how both the baseline reorder and TAG gather costs scale with payload, segment count, and
  MRAM locality. A descriptor removes host address-generation work, but does not make the bytes
  or gather operations free.
- **Deliverable:** a results table + the `off`-vs-`on` speedup on the reorder component;
  cross-checked against the measured pure-reorder fraction of end-to-end time.

---

## 5. The two pieces the TAG hardware spec must pin down

Everything above is buildable from what we know except two hardware-defining knobs, which the
TAG spec supplies and which are isolated behind `src/host/vm/tag/`:

1. **Descriptor / addressing model** — the exact compact descriptor format and the
   `TagAddrGen` function (does TAG walk the logical tensor and gather, or walk DPUs and scatter;
   what granularity; which of the 5 sites share one descriptor shape). Default assumption: it
   encodes the `q/k/v_offset` geometry directly.
2. **Timing model** — address-gen throughput, descriptor setup latency, and cross-DPU
   parallelism of the gather. These set `TagTransferCost`. Default assumption: one segment per
   cycle after a fixed setup, gathering across all DPUs in the rank in parallel.

Both are single files/functions; swapping the spec in does not touch Phases A/B/D.

---

## 6. Deliverables summary

| Phase | Adds | Key file(s) | Proves |
|-------|------|-------------|--------|
| A | readback costs cycles | `virtual_machine.go` (enable `SimulateMemory`) | transfer has a measurable cost |
| B | baseline reorder cost model | `src/host/vm/tag/pattern.go`, `cost.go` | isolated CPU reorder cost, in-sim |
| C | TAG engine + `--tag` switch | `src/host/vm/tag/engine.go`, `main.go` | third read method exists |
| D | reorder bit-match | native post-reorder tap, `cosim_check.py` | TAG is functionally exact |
| E | sweeps | scripts under `tools/` | measured TAG benefit and scaling |

The functional anchor throughout is the Phase-1 bit-match, now extended from DPU-tiled output to
post-reorder logical output — TAG is correct iff it reproduces PIM-DL's reorder byte-for-byte.
