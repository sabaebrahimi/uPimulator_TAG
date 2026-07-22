#!/usr/bin/env python3
"""Build hybrid co-sim stats: PIM-DL host + uPIMulator PIM."""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
from typing import Dict, List, Tuple

PROJS = ("qkv", "o", "ffn1", "ffn2")


def parse_cycles(log_path: str) -> Tuple[int, int]:
    logic = None
    memory = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"Logic\[[^\]]+\]_logic_cycle:\s*(\d+)", line)
            if m:
                logic = int(m.group(1))
            m = re.search(r"MemoryController\[[^\]]+\]_memory_cycle:\s*(\d+)", line)
            if m:
                memory = int(m.group(1))
    if logic is None or memory is None:
        raise ValueError(f"missing logic/memory cycle counters in {log_path}")
    return logic, memory


def parse_transfer_cycles(log_path: str) -> Dict[str, int]:
    """Parse HostTransfer_* stats from a co-sim log (0 if absent)."""
    out = {
        "transfer_cycle": 0,
        "h2d_cycle": 0,
        "d2h_cycle": 0,
        "h2d_bytes": 0,
        "d2h_bytes": 0,
    }
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            for key in out:
                m = re.search(rf"HostTransfer_{key}:\s*(\d+)", line)
                if m:
                    out[key] = int(m.group(1))
    return out


def parse_pimdl_single_layer(results_path: str) -> Dict[str, Dict[str, float | None]]:
    """
    Parse cumulative PIM-DL counters from `single_layer_results` and return
    average per-projection deltas (ms) for:
      - index_calc_ms
      - other_ms
      - data_transfer_ms
      - pim_kernel_ms
    """
    cumulative_index: List[float] = []
    cumulative_other: List[float] = []
    cumulative_data: List[float] = []
    cumulative_kernel: List[float] = []
    attn_reorder_iters: List[float] = []
    attn_cpu_iters: List[float] = []
    post_o_reorder_iters: List[float] = []
    gelu_ffn1_reorder_iters: List[float] = []
    post_ffn2_reorder_iters: List[float] = []
    alloc_overhead_iters: List[float] = []
    other_non_amm_iters: List[float] = []

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
            m = re.search(r"attention reorder time\s+([0-9]+\.[0-9]+),\s*attention cpu compute time\s+([0-9]+\.[0-9]+)", line)
            if m:
                attn_reorder_iters.append(float(m.group(1)))
                attn_cpu_iters.append(float(m.group(2)))
            m = re.search(
                r"post o reorder\+norm\+residual time\s+([0-9]+\.[0-9]+),\s*gelu\+ffn1 reorder time\s+([0-9]+\.[0-9]+),\s*post ffn2 reorder\+norm\+residual time\s+([0-9]+\.[0-9]+)",
                line,
            )
            if m:
                post_o_reorder_iters.append(float(m.group(1)))
                gelu_ffn1_reorder_iters.append(float(m.group(2)))
                post_ffn2_reorder_iters.append(float(m.group(3)))
            m = re.search(
                r"tensor allocation overhead time\s+([0-9]+\.[0-9]+),\s*other non amm time\s+([0-9]+\.[0-9]+)",
                line,
            )
            if m:
                alloc_overhead_iters.append(float(m.group(1)))
                other_non_amm_iters.append(float(m.group(2)))

    has_cumulative = bool(cumulative_index and cumulative_other and cumulative_data and cumulative_kernel)
    has_detailed = bool(attn_reorder_iters and post_o_reorder_iters and gelu_ffn1_reorder_iters and post_ffn2_reorder_iters)

    if not has_cumulative and not has_detailed:
        raise ValueError(
            "missing parsable host counters in PIM-DL results file. "
            "Need either cumulative counters (index/other/data transfer/pim kernel) "
            "or detailed breakdown lines (attention/post-o/ffn reorder timings)."
        )

    if not has_cumulative and has_detailed:
        # Fallback mode: detailed breakdown run (e.g., --need_breakdown --breakdown_type 1)
        # does not print cumulative index/data-transfer counters. We still provide
        # a projection-aligned host estimate from detailed host components.
        alloc_share = (statistics.mean(alloc_overhead_iters) / 4.0) if alloc_overhead_iters else 0.0
        other_share = (statistics.mean(other_non_amm_iters) / 4.0) if other_non_amm_iters else 0.0
        qkv_reorder = statistics.mean(attn_reorder_iters)
        qkv_cpu = statistics.mean(attn_cpu_iters) if attn_cpu_iters else 0.0
        o_reorder = statistics.mean(post_o_reorder_iters)
        ffn1_reorder = statistics.mean(gelu_ffn1_reorder_iters)
        ffn2_reorder = statistics.mean(post_ffn2_reorder_iters)

        # PIM-DL prints wall times in seconds; report columns are milliseconds.
        def s_to_ms(v: float) -> float:
            return v * 1000.0

        out_fallback: Dict[str, Dict[str, float | None]] = {
            "qkv": {
                "index_calc_ms": None,
                "other_ms": s_to_ms(qkv_cpu + alloc_share + other_share),
                "data_transfer_ms": None,
                "pim_kernel_ms": None,
                "reorder_ms": s_to_ms(qkv_reorder),
            },
            "o": {
                "index_calc_ms": None,
                "other_ms": s_to_ms(alloc_share + other_share),
                "data_transfer_ms": None,
                "pim_kernel_ms": None,
                "reorder_ms": s_to_ms(o_reorder),
            },
            "ffn1": {
                "index_calc_ms": None,
                "other_ms": s_to_ms(alloc_share + other_share),
                "data_transfer_ms": None,
                "pim_kernel_ms": None,
                "reorder_ms": s_to_ms(ffn1_reorder),
            },
            "ffn2": {
                "index_calc_ms": None,
                "other_ms": s_to_ms(alloc_share + other_share),
                "data_transfer_ms": None,
                "pim_kernel_ms": None,
                "reorder_ms": s_to_ms(ffn2_reorder),
            },
        }
        for p in PROJS:
            out_fallback[p]["host_total_ms"] = (
                (out_fallback[p]["reorder_ms"] or 0.0)
                + (out_fallback[p]["other_ms"] or 0.0)
            )
        print(
            "warning: cumulative host counters missing; using detailed breakdown fallback "
            "(projection-aligned host_total = reorder + shared host-overhead estimate)."
        )
        return out_fallback

    n = min(len(cumulative_index), len(cumulative_other), len(cumulative_data), len(cumulative_kernel))
    cumulative_index = cumulative_index[:n]
    cumulative_other = cumulative_other[:n]
    cumulative_data = cumulative_data[:n]
    cumulative_kernel = cumulative_kernel[:n]

    index_deltas: List[float] = []
    other_deltas: List[float] = []
    data_deltas: List[float] = []
    kernel_deltas: List[float] = []
    prev_i = prev_o = prev_d = prev_k = 0.0
    for i in range(n):
        index_deltas.append(cumulative_index[i] - prev_i)
        other_deltas.append(cumulative_other[i] - prev_o)
        data_deltas.append(cumulative_data[i] - prev_d)
        kernel_deltas.append(cumulative_kernel[i] - prev_k)
        prev_i = cumulative_index[i]
        prev_o = cumulative_other[i]
        prev_d = cumulative_data[i]
        prev_k = cumulative_kernel[i]

    usable = (n // 4) * 4
    if usable == 0:
        raise ValueError("need at least 4 cumulative latency points for qkv/o/ffn1/ffn2")
    index_deltas = index_deltas[:usable]
    other_deltas = other_deltas[:usable]
    data_deltas = data_deltas[:usable]
    kernel_deltas = kernel_deltas[:usable]

    by_proj_idx: Dict[str, List[float]] = {p: [] for p in PROJS}
    by_proj_oth: Dict[str, List[float]] = {p: [] for p in PROJS}
    by_proj_dat: Dict[str, List[float]] = {p: [] for p in PROJS}
    by_proj_ker: Dict[str, List[float]] = {p: [] for p in PROJS}
    for i in range(usable):
        p = PROJS[i % 4]
        by_proj_idx[p].append(index_deltas[i])
        by_proj_oth[p].append(other_deltas[i])
        by_proj_dat[p].append(data_deltas[i])
        by_proj_ker[p].append(kernel_deltas[i])

    out: Dict[str, Dict[str, float | None]] = {}
    # Optional detailed reorder metrics are printed once per full-layer iteration.
    # Map them by dominant projection path for hybrid per-projection comparison.
    reorder_map = {
        "qkv": statistics.mean(attn_reorder_iters) if attn_reorder_iters else None,
        "o": statistics.mean(post_o_reorder_iters) if post_o_reorder_iters else None,
        "ffn1": statistics.mean(gelu_ffn1_reorder_iters) if gelu_ffn1_reorder_iters else None,
        "ffn2": statistics.mean(post_ffn2_reorder_iters) if post_ffn2_reorder_iters else None,
    }
    # PIM-DL prints wall times in seconds; report columns are milliseconds.
    def s_to_ms(v: float) -> float:
        return v * 1000.0

    reorder_map_ms = {
        p: (s_to_ms(v) if v is not None else None) for p, v in reorder_map.items()
    }
    for p in PROJS:
        out[p] = {
            "index_calc_ms": s_to_ms(statistics.mean(by_proj_idx[p])),
            "other_ms": s_to_ms(statistics.mean(by_proj_oth[p])),
            "data_transfer_ms": s_to_ms(statistics.mean(by_proj_dat[p])),
            "pim_kernel_ms": s_to_ms(statistics.mean(by_proj_ker[p])),
            "reorder_ms": reorder_map_ms[p],
        }
        out[p]["host_total_ms"] = (
            out[p]["index_calc_ms"] + out[p]["other_ms"] + out[p]["data_transfer_ms"]
        )
    if not any(v is not None for v in reorder_map.values()):
        print(
            "warning: no detailed reorder lines found in PIM-DL results. "
            "Regenerate with --need_breakdown --breakdown_type 1 to populate "
            "reorder columns."
        )
    return out


def fmt(v: float) -> str:
    return f"{v:10.3f}"


def fmt_opt(v: float | None) -> str:
    if v is None:
        return f"{'-':>10}"
    return f"{v:10.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default=os.path.join("..", "cosim_logs"),
                    help="directory containing log_<proj>.txt files")
    ap.add_argument("--logic_mhz", type=float, default=350.0,
                    help="logic clock in MHz (default: 350)")
    ap.add_argument("--memory_mhz", type=float, default=2400.0,
                    help="memory clock in MHz (default: 2400)")
    ap.add_argument("--pimdl_results", required=True,
                    help="path to PIM-DL single_layer_results file for host-side breakdown")
    ap.add_argument("--csv", default="",
                    help="optional output CSV path")
    args = ap.parse_args()

    pimdl = parse_pimdl_single_layer(args.pimdl_results)

    rows = []
    for p in PROJS:
        lp = os.path.join(args.logs_dir, f"log_{p}.txt")
        if not os.path.exists(lp):
            raise FileNotFoundError(f"missing {lp}. Run run_cosim.sh first.")
        logic_c, mem_c = parse_cycles(lp)
        xfer = parse_transfer_cycles(lp)
        logic_ms = logic_c / (args.logic_mhz * 1000.0)
        mem_ms = mem_c / (args.memory_mhz * 1000.0)
        # Host↔MRAM transfer uses the same DRAM timing domain as DPU MRAM.
        xfer_ms = xfer["transfer_cycle"] / (args.memory_mhz * 1000.0)
        h2d_ms = xfer["h2d_cycle"] / (args.memory_mhz * 1000.0)
        d2h_ms = xfer["d2h_cycle"] / (args.memory_mhz * 1000.0)
        upim_pim_ms = max(logic_ms, mem_ms)
        host = pimdl[p]
        rows.append((
            p, logic_c, mem_c, logic_ms, mem_ms, upim_pim_ms,
            host["index_calc_ms"], host["other_ms"], host["data_transfer_ms"],
            host["host_total_ms"], host["pim_kernel_ms"], host["reorder_ms"],
            xfer["transfer_cycle"], xfer["h2d_cycle"], xfer["d2h_cycle"],
            xfer_ms, h2d_ms, d2h_ms,
            xfer["h2d_bytes"], xfer["d2h_bytes"],
        ))

    header = (
        "Projection      host_idx_ms host_other_ms host_xfer_ms host_total_ms "
        "host_reorder_ms reorder/host upim_logic_ms upim_mem_ms upim_pim_ms "
        "upim_xfer_ms upim_h2d_ms upim_d2h_ms host_vs_upim_ratio"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        (p, lc, mc, lms, mms, upim_pim_ms, idx, oth, xfer, host_total, pimdl_pim_ms,
         reorder_ms, xfer_c, h2d_c, d2h_c, xfer_ms, h2d_ms, d2h_ms, h2d_b, d2h_b) = r
        ratio = host_total / upim_pim_ms if upim_pim_ms > 0 else 0.0
        reorder_ratio = (reorder_ms / host_total) if (reorder_ms is not None and host_total > 0) else None
        print(
            f"{p:<10} {fmt_opt(idx)} {fmt_opt(oth)} {fmt_opt(xfer)} {fmt_opt(host_total)} "
            f"{fmt_opt(reorder_ms)} {fmt_opt(reorder_ratio)} {fmt(lms)} {fmt(mms)} {fmt(upim_pim_ms)} "
            f"{fmt(xfer_ms)} {fmt(h2d_ms)} {fmt(d2h_ms)} {ratio:16.3f}x"
        )

    tot_host = sum((r[9] or 0.0) for r in rows)
    tot_upim = sum(r[5] for r in rows)
    tot_xfer = sum(r[15] for r in rows)
    reorder_vals = [r[11] for r in rows if r[11] is not None]
    tot_reorder = sum(reorder_vals) if reorder_vals else None
    tot_reorder_ratio = (tot_reorder / tot_host) if (tot_reorder is not None and tot_host > 0) else None
    ratio_tot = tot_host / tot_upim if tot_upim > 0 else 0.0
    print("-" * len(header))
    print(
        f"{'TOTAL':<10} {'-':>10} {'-':>10} {'-':>10} {fmt(tot_host)} "
        f"{fmt_opt(tot_reorder)} {fmt_opt(tot_reorder_ratio)} "
        f"{'-':>10} {'-':>10} {fmt(tot_upim)} "
        f"{fmt(tot_xfer)} {'-':>10} {'-':>10} {ratio_tot:16.3f}x"
    )

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "projection",
                "pimdl_host_index_calc_ms",
                "pimdl_host_other_ms",
                "pimdl_host_data_transfer_ms",
                "pimdl_host_total_ms",
                "pimdl_host_reorder_ms",
                "pimdl_host_reorder_to_total_ratio",
                "pimdl_pim_kernel_ms_reference",
                "upim_logic_cycle",
                "upim_memory_cycle",
                "upim_logic_ms",
                "upim_memory_ms",
                "upim_pim_ms",
                "upim_transfer_cycle",
                "upim_h2d_cycle",
                "upim_d2h_cycle",
                "upim_transfer_ms",
                "upim_h2d_ms",
                "upim_d2h_ms",
                "upim_h2d_bytes",
                "upim_d2h_bytes",
                "host_vs_upim_ratio",
            ])
            for r in rows:
                (p, lc, mc, lms, mms, upim_pim_ms, idx, oth, xfer, host_total, pimdl_pim_ms,
                 reorder_ms, xfer_c, h2d_c, d2h_c, xfer_ms, h2d_ms, d2h_ms, h2d_b, d2h_b) = r
                ratio = host_total / upim_pim_ms if upim_pim_ms > 0 else 0.0
                reorder_ratio = (reorder_ms / host_total) if (reorder_ms is not None and host_total > 0) else ""
                w.writerow([
                    p,
                    idx if idx is not None else "",
                    oth if oth is not None else "",
                    xfer if xfer is not None else "",
                    host_total if host_total is not None else "",
                    reorder_ms if reorder_ms is not None else "",
                    reorder_ratio,
                    pimdl_pim_ms,
                    lc, mc, lms, mms, upim_pim_ms,
                    xfer_c, h2d_c, d2h_c, xfer_ms, h2d_ms, d2h_ms, h2d_b, d2h_b,
                    ratio,
                ])
            w.writerow([
                "TOTAL", "", "", "", tot_host, tot_reorder if tot_reorder is not None else "", tot_reorder_ratio if tot_reorder_ratio is not None else "",
                "",
                "", "", "", "", tot_upim,
                "", "", "", tot_xfer, "", "", "", "",
                ratio_tot,
            ])
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
