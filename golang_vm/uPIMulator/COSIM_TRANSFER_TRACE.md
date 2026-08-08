# Transfer trace replay

Set `COSIM_XFER_MODE=trace_cycle` to replay PIM-DL host↔MRAM transfer timing
through uPIMulator's memory controller. PIM-DL emits one trace per projection:

```text
pimdl_xfer_trace_qkv.jsonl
pimdl_xfer_trace_o.jsonl
pimdl_xfer_trace_ffn1.jsonl
pimdl_xfer_trace_ffn2.jsonl
```

For the normal four-projection co-simulation, `tools/run_cosim.sh` copies the
matching file from the dump directory into each patch directory and passes its
path to uPIMulator:

```bash
cd golang_vm/uPIMulator
COSIM_XFER_MODE=trace_cycle tools/run_cosim.sh /path/to/pimdl-dumps
```

The script maps `qkv`, `o`, `ffn1`, and `ffn2` to `PIMDL`, `PIMDL_O`,
`PIMDL_FFN1`, and `PIMDL_FFN2`, respectively. In `trace_cycle` mode,
preparation fails if the matching per-boundary trace is absent.

For a direct invocation, pass the exact boundary trace explicitly:

```bash
COSIM_XFER_MODE=trace_cycle build/uPIMulator \
  --benchmark PIMDL_O \
  --pimdl_patch_dirpath /path/to/patch \
  --pimdl_xfer_trace_path /path/to/pimdl_xfer_trace_o.jsonl
```

When `--pimdl_xfer_trace_path` is omitted, the PIM-DL benchmarks discover
`pimdl_xfer_trace_<boundary>.jsonl` in `--pimdl_patch_dirpath`. Other
benchmarks retain the legacy `pimdl_xfer_trace.jsonl` default. A missing trace
falls back to the existing PIM-DL transfer behavior.

The first JSONL line must be exactly a version-1 header:

```json
{"version":1,"type":"header","num_dpus":16}
```

Every following line must be a version-1 transfer event:

```json
{"version":1,"type":"transfer","boundary":"qkv","batch":0,"direction":"h2d","dpu":0,"symbol":"input_index","mram_offset":0,"bytes":8192,"host_offset":0,"kind":"input"}
```

Each trace must contain exactly one boundary; mixed-boundary input is rejected.
Batches are nondecreasing and may not mix boundaries or directions. H2D and D2H
batches may be interleaved: validation does not impose a global direction
ordering. The trace topology must match the simulator, symbols resolve to
absolute MRAM virtual addresses from `addresses.txt`, and every
symbol-relative MRAM range must be valid.

For H2D, the replayer copies the DPU's current MRAM range into a host arena,
checkpoints once per batch, and then runs every batch event in one
`SimulateMemory` call. This preserves functional MRAM contents while charging
cycle timing. D2H events drain into arena allocations after the launch.
