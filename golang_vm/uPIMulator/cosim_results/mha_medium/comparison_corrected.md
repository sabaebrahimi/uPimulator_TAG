# Corrected cycle-accurate comparison

Both scenarios use `trace_cycle`. PIM time is derived from kernel logic cycles; cumulative memory cycles are diagnostic only because transfer replay advances them.

| Scenario | DPUs | Host (ms) | Transfer (ms) | PIM (ms) | Reorder (ms) | Total (ms) |
|---|---:|---:|---:|---:|---:|---:|
| normal_small | 1 | 0.8123 | 0.1576 | 4.4215 | 0.0145 | 5.3915 |
| normal_small | 8 | 0.8921 | 0.1907 | 0.6799 | 0.0275 | 1.7627 |
| mha_medium | 1 | 1.4041 | 1.1191 | 200.4202 | 0.0220 | 202.9435 |
| mha_medium | 8 | 1.3776 | 1.0500 | 24.9873 | 0.0205 | 27.4149 |

| Scenario | PIM speedup | Transfer change, 1→8 | Reorder change, 1→8 | Total speedup |
|---|---:|---:|---:|---:|
| normal_small | 6.503× | +21.0% | +89.7% | 3.059× |
| mha_medium | 8.021× | -6.2% | -6.8% | 7.403× |

The original normal-layer report used cumulative memory cycles for PIM and therefore double-counted some transfer activity. Its original PIM values were 4.8147 ms and 1.7389 ms; the corrected values are 4.4215 ms and 0.6799 ms.

The multi-DPU uPIMulator timing launches all DPUs, while functional bit matching currently covers DPU 0’s shard.
