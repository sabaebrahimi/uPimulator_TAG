#!/usr/bin/env python3
"""Measure the PIM-DL host path by replaying saved DPU output shards."""

import argparse
import csv
import os
from pathlib import Path
import re
import statistics
import subprocess


def run(command, cwd, env):
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        print(result.stdout)
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.stdout


def medians(rows):
    return [statistics.median(row[column] for row in rows[-10:]) * 1000
            for column in range(len(rows[0]))]


def parse_host(text):
    detailed = [tuple(map(float, row)) for row in re.findall(
        r"reorder breakdown: qkv_to_attention ([\d.]+), o ([\d.]+), "
        r"ffn1 ([\d.]+), ffn2 ([\d.]+)", text)]
    host = [tuple(map(float, row)) for row in re.findall(
        r"host breakdown: attention ([\d.]+), reorder ([\d.]+), other ([\d.]+)", text)]
    indexes = [float(value) for value in re.findall(r"index calculation time ([\d.]+)", text)]
    amm_other = [float(value) for value in re.findall(r"other latency ([\d.]+),", text)]
    if len(detailed) < 10 or len(host) < 10 or len(indexes) < 44 or len(amm_other) < 44:
        raise ValueError("incomplete host replay output")

    qkv, o, ffn1, ffn2 = medians(detailed)
    attention, _, other = medians(host)
    # AMM counters are cumulative: four projections per layer iteration.
    other += (indexes[-1] - indexes[-41] + amm_other[-1] - amm_other[-41]) / 10 * 1000
    reorder = qkv + o + ffn1 + ffn2
    total = attention + reorder + other
    return {
        "host_ms": total,
        "attention_ms": attention,
        "qkv_to_attention_reorder_ms": qkv,
        "o_reorder_ms": o,
        "ffn1_reorder_ms": ffn1,
        "ffn2_reorder_ms": ffn2,
        "reorder_ms": reorder,
        "other_host_ms": other,
        "reorder_percent": 100 * reorder / total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upmem-home", type=Path,
                        default=Path("/home/saba/master/upmem-2025.1.0-Linux-x86_64"))
    parser.add_argument("--results-dir", type=Path, default=Path("cosim_results/host_only"))
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--runs", type=int, default=9,
                        help="interleaved host processes per layout (default: 9)")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    upim = Path(__file__).resolve().parents[1]
    repo = upim.parents[1]
    pimdl = repo / "PIM-DL-ASPLOS/inference-engine"
    results = (upim / args.results_dir).resolve()
    results.mkdir(parents=True, exist_ok=True)
    python_lib = Path("/home/saba/.pyenv/versions/3.10.16/lib")

    env = os.environ.copy()
    env.update({
        "UPMEM_HOME": str(args.upmem_home),
        "PATH": f"{args.upmem_home / 'bin'}:{env['PATH']}",
        "LD_LIBRARY_PATH": f"{args.upmem_home / 'lib'}:{python_lib}:{pimdl / 'build/lib'}",
        "LIBRARY_PATH": f"{args.upmem_home / 'lib'}:{python_lib}",
        "PKG_CONFIG_PATH": str(args.upmem_home / "share/pkgconfig"),
        "PIMDL_LAYER_CTX_MB": "64",
        "PIMDL_DATA_CTX_MB": "64",
    })
    if not args.no_build:
        run(["cmake", "-S", str(pimdl), "-B", str(pimdl / "build"),
             "-DLATENCY_BREAKDOWN=1", "-DLUT_BREAKDOWN=1"], repo, env)
        run(["cmake", "--build", str(pimdl / "build"), "--target",
             "test_transformer_layer", "-j"], repo, env)

    binary = pimdl / "build/bin/test_transformer_layer"
    outputs = {1: [], 8: []}
    for sample in range(args.runs):
        for dpus in (1, 8):
            print(f"[{dpus} DPU] host replay {sample + 1}/{args.runs}")
            config = pimdl / f"configs/cosim_results_{dpus}dpu.yaml"
            dumps = upim / f"cosim_results/{dpus}dpu/dumps"
            outputs[dpus].append(run([str(binary), str(config)], pimdl,
                                     env | {"PIMDL_HOST_REPLAY_DIR": str(dumps)}))

    rows = []
    for dpus in (1, 8):
        (results / f"host_{dpus}dpu.log").write_text("\n".join(outputs[dpus]), encoding="utf-8")
        samples = [parse_host(output) for output in outputs[dpus]]
        row = {key: statistics.median(sample[key] for sample in samples)
               for key in samples[0]}
        row["reorder_ms"] = sum(row[key] for key in (
            "qkv_to_attention_reorder_ms", "o_reorder_ms",
            "ffn1_reorder_ms", "ffn2_reorder_ms"))
        row["host_ms"] = row["attention_ms"] + row["reorder_ms"] + row["other_host_ms"]
        row["reorder_percent"] = 100 * row["reorder_ms"] / row["host_ms"]
        rows.append({"dpus": dpus, **row})

    csv_path = results / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows({key: value if key == "dpus" else f"{value:.6f}"
                          for key, value in row.items()} for row in rows)

    print("\nDPUs  Host(ms)  Attention  QKV->Attn  O reorder  FFN1  FFN2  Reorder(ms)  Weight")
    for row in rows:
        print(f"{row['dpus']:>4}  {row['host_ms']:>8.4f}  {row['attention_ms']:>9.4f}  "
              f"{row['qkv_to_attention_reorder_ms']:>9.4f}  {row['o_reorder_ms']:>9.4f}  "
              f"{row['ffn1_reorder_ms']:>5.4f}  {row['ffn2_reorder_ms']:>5.4f}  "
              f"{row['reorder_ms']:>11.4f}  {row['reorder_percent']:>6.2f}%")
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
