# Reorder analysis

## Conclusion

Pure CPU reordering is small for the configurations measured here, and increasing from 1
to 8 DPUs does not materially increase it when the feature microtile size is held constant.
Multi-head attention changes the destination address calculation, but it does not add reorder
work by itself. The amount of work is controlled mainly by tensor volume and microtile size.

## Measurements

The cycle-accurate rows use the corrected `trace_cycle` end-to-end results. The controlled
host replay rows are the median of seven interleaved processes, with each process reporting
the median of its final ten layer iterations. Host replay isolates the CPU layout loops and
does not include transfer or DPU execution.

| Experiment | DPUs | Pure reorder (ms) | 1-to-N change | Host (ms) | End-to-end (ms) |
|---|---:|---:|---:|---:|---:|
| Normal small, cycle-accurate | 1 | 0.0145 | baseline | 0.8123 | 5.3915 |
| Normal small, cycle-accurate | 8 | 0.0275 | +89.7% | 0.8921 | 1.7627 |
| Medium MHA, cycle-accurate | 1 | 0.0220 | baseline | 1.4041 | 202.9435 |
| Medium MHA, cycle-accurate | 8 | 0.0205 | -6.8% | 1.3776 | 27.4149 |
| Medium MHA, controlled host replay | 1 | 0.0145 | baseline | — | — |
| Medium MHA, controlled host replay | 8 | 0.0155 | +6.9% (+0.0010 ms) | — | — |
| Full MHA, controlled host replay | 1 | 0.2225 | baseline | 20.9099 | — |
| Full MHA, controlled host replay | 8 | 0.2245 | +0.9% (+0.0020 ms) | 20.6130 | — |
| Full MHA, controlled host replay | 16 | 0.2600 | +16.9% (+0.0375 ms) | 20.6123 | — |

The medium host-only samples were tightly grouped: 0.0140–0.0150 ms at 1 DPU and
0.0150–0.0160 ms at 8 DPUs. The 1 μs difference is too small to support a claim that eight
DPUs cause a meaningful reorder penalty.

## Why 1 and 8 DPUs are nearly identical

For a projection with `F` output features, `n` rows, `L` LUT-parallel DPUs, and microtile
width `m`, each DPU owns `F/L` features. The aggregate number of microtile blocks is:

```
L * n * (F / L) / m = n * F / m
```

The DPU count cancels. Eight DPUs create more shards, but each shard contains one eighth as
many features. The implementation still moves the same bytes through the same number of
microtile blocks.

| Configuration | DPUs | Reordered payload/layer | Microtiles (QKV/O/FFN1/FFN2) | Aggregate block iterations |
|---|---:|---:|---|---:|
| Normal small | 1 | 56 KiB | 16 / 16 / 16 / 16 | 896 |
| Normal small | 8 | 56 KiB | 4 / 4 / 8 / 4 | 3,072 |
| Medium MHA | 1 | 224 KiB | 16 / 16 / 16 / 16 | 3,584 |
| Medium MHA | 8 | 224 KiB | 16 / 16 / 16 / 16 | 3,584 |
| Full MHA | 1 | 3,456 KiB | 32 / 32 / 32 / 32 | 27,648 |
| Full MHA | 8 | 3,456 KiB | 32 / 32 / 32 / 32 | 27,648 |
| Full MHA | 16 | 3,456 KiB | 16 / 16 / 16 / 16 | 55,296 |

This also explains the old normal-layer jump. Its 8-DPU configuration reduced the
microtiles, increasing the number of short copy/loop blocks by 3.43x. The DPU count was
confounded with microtile fragmentation. In the full MHA configuration, 16 DPUs likewise
halve the microtile width and double the block count, which produces the modest measured
increase.

The MHA head count appears only in the destination indexing. The parser requires the head
dimension and every per-DPU Q/KV shard to be divisible by the microtile width. Consequently,
a block never crosses a head or shard boundary in these tests. Increasing the head count at
fixed token width therefore changes where bytes land, not how many bytes or blocks are moved.
If the token width grows with the head count, absolute reorder time grows with that larger
tensor; the full MHA result demonstrates that volume effect.

## Timer scope

`reorder_ms` is the sum of the four explicitly timed layout loops: QKV-to-attention, O,
FFN1, and FFN2. Earlier project notes combined four larger stages and called their sum
reordering. Those stages also contain GELU, normalization, and residual operations, so they
are only a reorder-related stage upper bound and are not work that a transfer-time gather can
remove.

| Controlled replay | DPUs | Pure reorder (ms) | Entire reorder-related stages (ms) |
|---|---:|---:|---:|
| Medium MHA | 1 | 0.0145 | 0.0505 |
| Medium MHA | 8 | 0.0155 | 0.0520 |
| Full MHA | 1 | 0.2225 | 0.6730 |
| Full MHA | 8 | 0.2245 | 0.6750 |
| Full MHA | 16 | 0.2600 | 0.7245 |

GQA is not covered by these measurements. The current attention implementation asserts
that `head_num == kv_head_num`, so a GQA configuration cannot execute the attention stage
yet. GQA should be measured only after that mapping is implemented and validated.

Longer-sequence validation and larger-layer estimates are in
`../mha_prediction/prediction.md`. Those measurements show a cache-size transition: the
effective reordered-payload rate falls from 15.8 GB/s at 128 tokens to 9.4 GB/s at 512 and
7.7 GB/s at 1024. Consequently, real-world prediction must account for sequence length and
cannot use one constant bandwidth derived from the small tensors.
