#!/usr/bin/env python3
"""Summarize 16-DPU co-sim timings: PIM kernel, host↔DPU transfer, host-side."""

from __future__ import annotations

import argparse
import os
import re
import statistics
from typing import Dict, List, Optional, Tuple

PROJS = ("qkv", "o", "ffn1", "ffn2")
LOGIC_MHZ = 350.0
MEM_MHZ = 2400.0
# Paper / upmem_reg_model fixed-BW model (GB/s per DPU, parallel xfer wall ≈ size/BW).
H2D_GBPS = 0.2957
D2H_GBPS = 0.0627

# Per-DPU shard sizes for cosim_static_small (must match cosim_prepare.py).
SHARDS = {
    "qkv":  {"lut": 32 * 48 * 16, "index": 64 * 32 * 2, "output": 64 * 48 * 4},
    "o":    {"lut": 32 * 16 * 16, "index": 64 * 32 * 2, "output": 64 * 16 * 4},
    "ffn1": {"lut": 32 * 32 * 16, "index": 64 * 32 * 2, "output": 64 * 32 * 4},
    "ffn2": {"lut": 64 * 16 * 16, "index": 64 * 64 * 2, "output": 64 * 16 * 4},
}


def cycles_to_ms(cycles: int, mhz: float) -> float:
    return cycles / (mhz * 1000.0)


def fixed_bw_ms(nbytes: int, gbps: float) -> float:
    return (nbytes * 1e3) / (gbps * (2**30))


def parse_log(path: str) -> Dict[str, int]:
    out: Dict[str, int] = {
        "logic_cycle": 0,
        "memory_cycle": 0,
        "transfer_cycle": 0,
        "h2d_cycle": 0,
        "d2h_cycle": 0,
        "h2d_bytes": 0,
        "d2h_bytes": 0,
        "num_dpus_seen": 0,
    }
    logic_vals: List[int] = []
    mem_vals: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"Logic\[[^\]]+\]_logic_cycle:\s*(\d+)", line)
            if m:
                logic_vals.append(int(m.group(1)))
            m = re.search(r"MemoryController\[[^\]]+\]_memory_cycle:\s*(\d+)", line)
            if m:
                mem_vals.append(int(m.group(1)))
            for key in ("transfer_cycle", "h2d_cycle", "d2h_cycle", "h2d_bytes", "d2h_bytes"):
                m = re.search(rf"HostTransfer_{key}:\s*(\d+)", line)
                if m:
                    out[key] = int(m.group(1))
    if logic_vals:
        out["logic_cycle"] = max(logic_vals)  # parallel DPUs: wall ≈ max
        out["num_dpus_seen"] = len(logic_vals)
    if mem_vals:
        out["memory_cycle"] = max(mem_vals)
    return out


def parse_pimdl_host(results_path: str) -> Optional[Dict[str, Dict[str, Optional[float]]]]:
    """Best-effort parse of PIM-DL native cumulative / detailed host timers."""
    if not results_path or not os.path.exists(results_path):
        return None

    cumulative_index: List[float] = []
    cumulative_other: List[float] = []
    cumulative_data: List[float] = []
    cumulative_kernel: List[float] = []
    attn_reorder: List[float] = []
    attn_cpu: List[float] = []
    post_o: List[float] = []
    gelu_ffn1: List[float] = []
    post_ffn2: List[float] = []

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"index calculation time\s+([0-9]+\.[0-9]+)", line)
            if m:
                cumulative_index.append(float(m.group(1)))
            m = re.search(
                r"other latency\s+([0-9]+\.[0-9]+),\s*data transfer latency\s+([0-9]+\.[0-9]+),\s*pim kernel latency\s+([0-9]+\.[0-9]+)",
                line,
            )
            if m:
                cumulative_other.append(float(m.group(1)))
                cumulative_data.append(float(m.group(2)))
                cumulative_kernel.append(float(m.group(3)))
            m = re.search(
                r"attention reorder time\s+([0-9]+\.[0-9]+),\s*attention cpu compute time\s+([0-9]+\.[0-9]+)",
                line,
            )
            if m:
                attn_reorder.append(float(m.group(1)))
                attn_cpu.append(float(m.group(2)))
            m = re.search(
                r"post o reorder\+norm\+residual time\s+([0-9]+\.[0-9]+),\s*gelu\+ffn1 reorder time\s+([0-9]+\.[0-9]+),\s*post ffn2 reorder\+norm\+residual time\s+([0-9]+\.[0-9]+)",
                line,
            )
            if m:
                post_o.append(float(m.group(1)))
                gelu_ffn1.append(float(m.group(2)))
                post_ffn2.append(float(m.group(3)))

    def mean_delta(xs: List[float]) -> Optional[List[float]]:
        if len(xs) < 5:
            return None
        # cumulative over 4 projections per layer iteration; take last full layer deltas
        deltas = []
        for i in range(1, len(xs)):
            deltas.append(xs[i] - xs[i - 1])
        # group into 4-tuples (qkv,o,ffn1,ffn2)
        if len(deltas) < 4:
            return None
        # use mean of each position across complete layers
        n_layers = len(deltas) // 4
        if n_layers == 0:
            return None
        out = []
        for k in range(4):
            vals = [deltas[i * 4 + k] for i in range(n_layers)]
            out.append(statistics.mean(vals) * 1000.0)  # s -> ms
        return out

    idx = mean_delta(cumulative_index)
    oth = mean_delta(cumulative_other)
    dat = mean_delta(cumulative_data)
    ker = mean_delta(cumulative_kernel)

    reorder = {
        "qkv": statistics.mean(attn_reorder) * 1000.0 if attn_reorder else None,
        "o": statistics.mean(post_o) * 1000.0 if post_o else None,
        "ffn1": statistics.mean(gelu_ffn1) * 1000.0 if gelu_ffn1 else None,
        "ffn2": statistics.mean(post_ffn2) * 1000.0 if post_ffn2 else None,
    }
    attn = statistics.mean(attn_cpu) * 1000.0 if attn_cpu else None

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for i, p in enumerate(PROJS):
        out[p] = {
            "index_ms": idx[i] if idx else None,
            "other_ms": oth[i] if oth else None,
            "data_transfer_ms": dat[i] if dat else None,
            "pim_kernel_ms": ker[i] if ker else None,
            "reorder_ms": reorder[p],
            "attn_cpu_ms": attn if p == "qkv" else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="cosim_logs")
    ap.add_argument("--pimdl_results", default="",
                    help="optional PIM-DL native results for host-side timers")
    ap.add_argument("--num_dpus", type=int, default=16)
    args = ap.parse_args()

    host = parse_pimdl_host(args.pimdl_results) if args.pimdl_results else None

    print(f"Co-sim timing report ({args.num_dpus} DPUs)")
    print(f"  PIM clocks: logic={LOGIC_MHZ} MHz, memory={MEM_MHZ} MHz")
    print(f"  Classic xfer BW: H2D={H2D_GBPS} GB/s, D2H={D2H_GBPS} GB/s (per-DPU parallel)")
    if host:
        print(f"  Host timers from: {args.pimdl_results}")
    else:
        print("  Host timers: (none — pass --pimdl_results for native host breakdown)")
    print()

    header = (
        f"{'Proj':<6} {'PIM_logic_ms':>12} {'PIM_mem_ms':>11} {'PIM_ms':>9} "
        f"{'Xfer_sim_ms':>11} {'H2D_sim':>9} {'D2H_sim':>9} "
        f"{'Xfer_BW_ms':>10} {'Host_idx':>9} {'Host_oth':>9} {'Host_reo':>9} {'Host_xfer':>9}"
    )
    print(header)
    print("-" * len(header))

    tot = {k: 0.0 for k in ("pim", "xfer_sim", "xfer_bw", "host")}
    for p in PROJS:
        lp = os.path.join(args.logs_dir, f"log_{p}.txt")
        if not os.path.exists(lp):
            print(f"{p:<6} MISSING {lp}")
            continue
        st = parse_log(lp)
        logic_ms = cycles_to_ms(st["logic_cycle"], LOGIC_MHZ)
        mem_ms = cycles_to_ms(st["memory_cycle"], MEM_MHZ)
        pim_ms = max(logic_ms, mem_ms)
        xfer_sim = cycles_to_ms(st["transfer_cycle"], MEM_MHZ)
        h2d_sim = cycles_to_ms(st["h2d_cycle"], MEM_MHZ)
        d2h_sim = cycles_to_ms(st["d2h_cycle"], MEM_MHZ)

        shard = SHARDS[p]
        # Parallel xfer: wall time ≈ per-DPU size / BW (all DPUs same shard size).
        h2d_bw = fixed_bw_ms(shard["lut"] + shard["index"], H2D_GBPS)
        d2h_bw = fixed_bw_ms(shard["output"], D2H_GBPS)
        xfer_bw = h2d_bw + d2h_bw

        h = host[p] if host else {}
        idx = h.get("index_ms") if h else None
        oth = h.get("other_ms") if h else None
        reo = h.get("reorder_ms") if h else None
        hx = h.get("data_transfer_ms") if h else None

        def fmt(v: Optional[float]) -> str:
            return f"{v:9.3f}" if v is not None else f"{'—':>9}"

        print(
            f"{p:<6} {logic_ms:12.3f} {mem_ms:11.3f} {pim_ms:9.3f} "
            f"{xfer_sim:11.3f} {h2d_sim:9.3f} {d2h_sim:9.3f} "
            f"{xfer_bw:10.3f} {fmt(idx)} {fmt(oth)} {fmt(reo)} {fmt(hx)}"
        )
        print(
            f"       bytes: h2d={st['h2d_bytes']} d2h={st['d2h_bytes']} "
            f"dpus_in_log={st['num_dpus_seen']} "
            f"shard(lut+idx+out)={shard['lut']+shard['index']+shard['output']}"
        )
        tot["pim"] += pim_ms
        tot["xfer_sim"] += xfer_sim
        tot["xfer_bw"] += xfer_bw

    print("-" * len(header))
    print(
        f"{'TOTAL':<6} {'':>12} {'':>11} {tot['pim']:9.3f} "
        f"{tot['xfer_sim']:11.3f} {'':>9} {'':>9} "
        f"{tot['xfer_bw']:10.3f}"
    )
    print()
    print("Notes:")
    print("  PIM_ms      = max(logic_ms, mem_ms) across DPUs (parallel launch).")
    print("  Xfer_sim_ms = HostTransfer_transfer_cycle @ memory clock (cycle-level model).")
    print("  Xfer_BW_ms  = classic uPIMulator size/BW model (paper Table I).")
    print("  Host_*      = PIM-DL native wall timers (ms); only valid if results match this config.")


if __name__ == "__main__":
    main()
