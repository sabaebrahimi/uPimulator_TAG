# Larger-MHA reorder prediction

## Scope

This model predicts the four pure CPU reorder loops for a single MHA layer. It does not
predict DPU kernel, transfer, attention-compute, or total layer latency. The prediction is for
the current PIM-DL implementation with aligned microtiles and a conventional FFN width of
`F = 4D`.

For batch `B`, sequence length `S`, hidden width `D`, and FFN width `F`, the reordered
payload is:

```
payload_bytes = 4 * B * S * (5D + F)
              = 36 * B * S * D       when F = 4D
```

Head count is absent from this equation because, for MHA, `D = heads * head_dim`. With the
alignment constraints used here, changing the number of heads at fixed `D` changes address
mapping but not payload or microtile-block count.

## Measured sequence-length scaling

These are controlled host replays of a 12-head, `D=768`, `F=3072`, batch-1 MHA layer. The
512- and 1024-token points were added specifically to validate the extrapolation. Times are
medians across interleaved 1- and 8-DPU layouts; the individual layouts remain within about
2% of one another.

| Sequence | Payload | 1 DPU (ms) | 8 DPUs (ms) | Mean (ms) | Effective payload rate |
|---:|---:|---:|---:|---:|---:|
| 128 | 3.375 MiB | 0.2225 | 0.2245 | 0.2235 | 15.8 GB/s |
| 512 | 13.5 MiB | 1.5085 | 1.4905 | 1.4995 | 9.4 GB/s |
| 1024 | 27 MiB | 3.6790 | 3.6495 | 3.6643 | 7.7 GB/s |

A constant-bandwidth prediction from the two smaller original points would predict only
0.89 ms at 512 tokens. The measured 1.50 ms shows that such a model is too optimistic once
the layout working set leaves cache. Most of the superlinear growth is in QKV-to-attention:
its V conversion writes with a `seq_len` stride.

## Predictions

For longer sequences at `D=768`, the lower bound assumes the 1024-token effective rate
remains constant. The upper bound continues the measured 512-to-1024 scaling exponent of
1.289. Width and batch projections scale this reference according to `B * (5D+F)`.

| Representative MHA layer | Payload/layer | Predicted pure reorder/layer |
|---|---:|---:|
| `B=1, S=2048, D=768, F=3072` | 54 MiB | 7.3–9.0 ms |
| `B=1, S=4096, D=768, F=3072` | 108 MiB | 14.7–21.9 ms |
| `B=1, S=512, D=1024, F=4096` | 18 MiB | about 2.0 ms |
| `B=1, S=2048, D=4096, F=16384` | 288 MiB | 39–48 ms |
| `B=8, S=512, D=768, F=3072` | 108 MiB | about 12–15 ms |
| `B=64, S=512, D=768, F=3072` | 864 MiB | about 96–120 ms |

The 1- and 8-DPU predictions are the same to useful precision as long as both layouts retain
the same microtile width. The measured 16-DPU full-MHA layout uses 16-float instead of
32-float microtiles and costs 16.9% more, so applying an approximately 1.17 multiplier is a
reasonable first estimate for that exact 16-DPU tiling choice.

The ranges are engineering estimates rather than confidence intervals. Extrapolation is
strongest along sequence length for `D=768`, where three measured sizes now exist. The
`D=4096` and large-batch rows also extrapolate width or batch and should be validated with a
host replay before being used as final paper numbers.

