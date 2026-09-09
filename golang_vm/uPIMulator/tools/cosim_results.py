#!/usr/bin/env python3
"""Run one PIM-DL layer on 1/8 DPUs and report the COSIM latency split."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import types

import yaml

PROJECTIONS = {
    "qkv": ("PIMDL", lambda n: n["token_dim"] + 2 * n.get("kv_head_num", n["head_num"]) * n["head_dim"]),
    "o": ("PIMDL_O", lambda n: n["token_dim"]),
    "ffn1": ("PIMDL_FFN1", lambda n: n["ffn_hidden_dim"]),
    "ffn2": ("PIMDL_FFN2", lambda n: n["token_dim"]),
}
FLOAT_BYTES = 4
LOGIC_MHZ = 350.0
MEMORY_MHZ = 2400.0


def run(command, cwd, env, *, capture=True):
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None)
    if result.returncode:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.stdout or ""


def configuration(path):
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    n = cfg["network_params"]
    k = cfg["kernel_params"]
    dpus = cfg["system_params"]["dpu_num"]
    if dpus not in (1, 8, 16):
        raise ValueError(f"{path}: expected 1, 8, or 16 DPUs, got {dpus}")
    for name in PROJECTIONS:
        if k[f"{name}_input_parallelism"] * k[f"{name}_lut_parallelism"] != dpus:
            raise ValueError(f"{path}: {name} parallelism does not equal {dpus} DPUs")
    return cfg


def geometry(cfg, projection):
    n = cfg["network_params"]
    k = cfg["kernel_params"]
    output_features = PROJECTIONS[projection][1](n)
    input_parallelism = k[f"{projection}_input_parallelism"]
    lut_parallelism = k[f"{projection}_lut_parallelism"]
    return {
        "n_stile": n["seq_len"] * n["batch_size"] // input_parallelism,
        "feature_stile": output_features // lut_parallelism,
        "n_mtile": k[f"{projection}_n_mtile_size"],
        "feature_mtile": k[f"{projection}_feature_mtile_size"],
        "cb_mtile": k[f"{projection}_cb_mtile_size"],
        "codebooks": k[f"{projection}_num_codebook"],
        "centroids": k[f"{projection}_num_centroid"],
    }


def build_pimdl(pimdl, config, dump_dir, env):
    sys.path.insert(0, str(pimdl))
    import run_layer

    args = types.SimpleNamespace(transformer_config_file=str(config), need_compile=True,
                                 need_measure_energy=False, need_breakdown=True,
                                 breakdown_type=1)
    old_cwd = os.getcwd()
    old_env = os.environ.copy()
    try:
        os.environ.update(env)
        os.chdir(pimdl)
        run_layer.compile(args)
    finally:
        os.chdir(old_cwd)
        os.environ.clear()
        os.environ.update(old_env)
    build = pimdl / "build"
    run(["cmake", "-DPIMDL_COSIM_DUMP=ON", "-DLATENCY_BREAKDOWN=1",
         "-DLUT_BREAKDOWN=1", str(pimdl)], build, env)
    run(["make", "-j"], build, env)
    binary = build / "bin/test_transformer_layer"

    dump_env = env | {"PIMDL_COSIM_DUMP_DIR": str(dump_dir), "PIMDL_COSIM_DUMP_ONLY": "1"}
    run([str(binary), str(config)], pimdl, dump_env)
    return run([str(binary), str(config)], pimdl,
               env | {"PIMDL_COSIM_DUMP_DIR": str(dump_dir)})


def build_upim_kernels(upim, cfg, build_dir, upmem_home, env):
    clang = upmem_home / "bin/dpu-upmem-dpurte-clang"
    support = upim / "benchmark/PIMDL/support"
    for projection, (benchmark, _) in PROJECTIONS.items():
        g = geometry(cfg, projection)
        output = build_dir / benchmark / "dpu/CMakeFiles" / f"{benchmark}_device.dir/task.c.o"
        output.parent.mkdir(parents=True, exist_ok=True)
        tasklets = int(env.get("PIMDL_NUM_TASKLETS", "16"))
        defines = {
            "NR_TASKLETS": tasklets,
            "N_STILE_SIZE": g["n_stile"],
            "FEATURE_STILE_SIZE": g["feature_stile"],
            "N_MTILE_SIZE": g["n_mtile"],
            "FEATURE_MTILE_SIZE": g["feature_mtile"],
            # This must match the host-side index tiling. LUT_CACHE_CB controls
            # the smaller independent MRAM-to-WRAM streaming granularity.
            "CB_MTILE_SIZE": g["cb_mtile"],
            "NUM_CODEBOOK": g["codebooks"],
            "NUM_CENTROID": g["centroids"],
            "LUT_LOAD_TYPE": cfg["kernel_params"][f"{projection}_lut_load_type"],
            "FEATURE_LOAD_TILE_SIZE": cfg["kernel_params"][f"{projection}_feature_load_tile_size"],
            "CB_LOAD_TILE_SIZE": cfg["kernel_params"][f"{projection}_cb_load_tile_size"],
        }
        command = [str(clang), "-w", "-I", str(support), "-Os", "-S"]
        command += [f"-D{name}={value}" for name, value in defines.items()]
        command += [str(upim / f"benchmark/{benchmark}/dpu/task.c"), "-o", str(output)]
        run(command, upim, env)


def stage_patch(cfg, projection, dump_dir, patch_dir):
    g = geometry(cfg, projection)
    tag = f"{projection}_cb{g['codebooks']}_fs{g['feature_stile']}"
    segments = patch_dir / "pimdl_segments"
    segments.mkdir(parents=True, exist_ok=True)
    for symbol, prefix in (("lut_table", "lut"), ("input_index", "index")):
        source = dump_dir / f"{prefix}_{tag}.bin"
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copyfile(source, segments / f"{symbol}.bin")
    manifest = {"anchor": "MRAM_BASE", "segments": [
        {"symbol": "lut_table", "path": "pimdl_segments/lut_table.bin"},
        {"symbol": "input_index", "path": "pimdl_segments/input_index.bin"},
    ]}
    (patch_dir / "pimdl_mram_patch.json").write_text(json.dumps(manifest), encoding="utf-8")
    trace = dump_dir / f"pimdl_xfer_trace_{projection}.jsonl"
    shutil.copyfile(trace, patch_dir / trace.name)
    return tag, g["n_stile"] * g["feature_stile"] * FLOAT_BYTES, patch_dir / trace.name


def parse_upim_log(text):
    def maximum(pattern):
        values = [int(value) for value in re.findall(pattern, text)]
        if not values:
            raise ValueError(f"missing {pattern} in uPIMulator log")
        return max(values)

    logic = maximum(r"Logic\[[^]]+\]_logic_cycle:\s*(\d+)")
    memory = maximum(r"MemoryController\[[^]]+\]_memory_cycle:\s*(\d+)")
    transfer = maximum(r"HostTransfer_transfer_cycle:\s*(\d+)")
    models = re.findall(r"HostTransfer_xfer_model_([a-z_]+):\s*[1-9]\d*", text)
    return {
        "logic_cycles": logic,
        "memory_cycles": memory,
        "transfer_cycles": transfer,
        # Logic cycles are the kernel wall clock: DpuComputeCycleJob keeps
        # cycling until both logic and its DPU-local memory controller drain.
        # The memory-cycle counter is cumulative and trace_cycle also advances
        # it during H2D/D2H replay, so using it here double-counts transfers.
        "pim_ms": logic / (LOGIC_MHZ * 1000),
        "transfer_ms": transfer / (MEMORY_MHZ * 1000),
        "xfer_model": "+".join(sorted(set(models))) if models else "unknown",
    }


def run_upim(upim, cfg, dump_dir, result_dir, build_dir, env):
    dpus = cfg["system_params"]["dpu_num"]
    tasklets = int(env.get("PIMDL_NUM_TASKLETS", "16"))
    pim_ms = transfer_ms = 0.0
    mismatches = []
    projection_rows = []
    bin_dir = result_dir / "bin_cosim"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for projection, (benchmark, _) in PROJECTIONS.items():
        patch = result_dir / f"patch_{projection}"
        tag, output_bytes, trace = stage_patch(cfg, projection, dump_dir, patch)
        command = [str(upim / "build/uPIMulator"), "--benchmark", benchmark,
                   "--root_dirpath", str(upim), "--bin_dirpath", str(bin_dir),
                   "--pimdl_patch_dirpath", str(patch), "--pimdl_xfer_trace_path", str(trace),
                   "--num_channels", "1", "--num_ranks_per_channel", "1",
                   "--num_dpus_per_rank", str(dpus), "--num_tasklets", str(tasklets),
                   "--skip_compile", "1", "--verbose", "0"]
        sim_env = env | {"UPIM_BENCH_BUILD_DIR": str(build_dir),
                         "COSIM_XFER_MODE": env.get("COSIM_XFER_MODE", "trace_cycle"),
                         "PIMDL_OUTPUT_BYTES": str(output_bytes)}
        run(command, upim, sim_env)
        log = (bin_dir / "log.txt").read_text(encoding="utf-8")
        (result_dir / f"upim_{projection}.log").write_text(log, encoding="utf-8")
        timing = parse_upim_log(log)
        pim_ms += timing["pim_ms"]
        transfer_ms += timing["transfer_ms"]
        actual = (bin_dir / "pimdl_output.bin").read_bytes()
        (result_dir / f"actual_{projection}.bin").write_bytes(actual)
        expected = (dump_dir / f"output_{tag}.bin").read_bytes()
        if actual != expected:
            mismatches.append(projection)
        projection_rows.append({
            "dpus": dpus,
            "projection": projection,
            "logic_cycles": timing["logic_cycles"],
            "memory_cycles": timing["memory_cycles"],
            "pim_ms": timing["pim_ms"],
            "transfer_cycles": timing["transfer_cycles"],
            "transfer_ms": timing["transfer_ms"],
            "xfer_model": timing["xfer_model"],
            "validated_output": "all_dpus" if dpus == 1 else "dpu0_shard",
        })
    if mismatches:
        raise ValueError(f"{dpus}-DPU output mismatch: {', '.join(mismatches)}")
    return pim_ms, transfer_ms, projection_rows


def parse_host(text):
    breakdown = [(float(a), float(r), float(o)) for a, r, o in re.findall(
        r"host breakdown: attention ([\d.]+), reorder ([\d.]+), other ([\d.]+)", text)]
    indexes = [float(v) for v in re.findall(r"index calculation time ([\d.]+)", text)]
    amm_other = [float(v) for v in re.findall(r"other latency ([\d.]+),", text)]
    if len(breakdown) < 10 or len(indexes) < 44 or len(amm_other) < 44:
        raise ValueError("incomplete PIM-DL host timing output")
    attention, reorder, other = (statistics.median(row[i] for row in breakdown[-10:]) * 1000
                                 for i in range(3))
    # Cumulative AMM counters: delta over the final ten four-projection iterations.
    index_ms = (indexes[-1] - indexes[-41]) / 10 * 1000
    amm_other_ms = (amm_other[-1] - amm_other[-41]) / 10 * 1000
    other += index_ms + amm_other_ms
    return attention, reorder, other


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upmem-home", type=Path,
                        default=Path("/home/saba/master/upmem-2025.1.0-Linux-x86_64"))
    parser.add_argument("--results-dir", type=Path, default=Path("cosim_results"))
    parser.add_argument("--resume", action="store_true",
                        help="reuse existing PIM-DL host logs and compiled DPU assembly")
    parser.add_argument("--model", choices=("tiny", "mha"), default="tiny")
    parser.add_argument("--config", type=Path,
                        help="use one explicit configuration (requires one --dpus value)")
    parser.add_argument("--dpus", type=int, choices=(1, 8, 16), action="append",
                        help="DPU count to run; repeat for multiple counts")
    args = parser.parse_args()
    if args.config and (not args.dpus or len(args.dpus) != 1):
        parser.error("--config requires exactly one --dpus value")

    upim = Path(__file__).resolve().parents[1]
    repo = upim.parents[1]
    pimdl = repo / "PIM-DL-ASPLOS/inference-engine"
    args.results_dir = (upim / args.results_dir).resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    py310 = Path(os.environ.get("PIMDL_PYTHON_LIB", "/home/saba/.pyenv/versions/3.10.16/lib"))
    if not (py310 / "libpython3.10.so.1.0").exists():
        fallback = Path("/home/saba/.pyenv/versions/3.10.14/lib")
        if (fallback / "libpython3.10.so.1.0").exists():
            py310 = fallback
    env = os.environ.copy()
    env.update({
        "UPMEM_HOME": str(args.upmem_home),
        "UPMEM_PROFILE_BASE": "backend=simulator",
        "PATH": f"{args.upmem_home / 'bin'}:{env['PATH']}",
        "LD_LIBRARY_PATH": f"{args.upmem_home / 'lib'}:{py310}:{pimdl / 'build/lib'}",
        "LIBRARY_PATH": f"{args.upmem_home / 'lib'}:{py310}",
        "PKG_CONFIG_PATH": str(args.upmem_home / "share/pkgconfig"),
        "PIMDL_LAYER_CTX_MB": "256" if args.model == "mha" else "64",
        "PIMDL_DATA_CTX_MB": "64",
    })
    run(["go", "build", "-buildvcs=false", "-o", "build/uPIMulator", "./src"], upim,
        env | {"GOCACHE": f"/tmp/cosim-go-cache-{os.getuid()}"})

    rows = []
    projection_rows = []
    default_dpus = (1, 8, 16) if args.model == "mha" else (1, 8)
    dpus_list = tuple(args.dpus) if args.dpus else default_dpus
    for dpus in dpus_list:
        if args.config:
            source_config = args.config.resolve()
            with source_config.open(encoding="utf-8") as stream:
                selected_cfg = yaml.safe_load(stream)
            if selected_cfg["system_params"]["dpu_num"] == dpus:
                config = source_config
            else:
                config = args.results_dir / f"{source_config.stem}_{dpus}dpu.yaml"
                selected_cfg["system_params"]["dpu_num"] = dpus
                for name in PROJECTIONS:
                    selected_cfg["kernel_params"][f"{name}_lut_parallelism"] = dpus
                config.write_text(yaml.safe_dump(selected_cfg, sort_keys=False), encoding="utf-8")
        elif args.model == "mha" and dpus == 16:
            # Keep the checked-in 1/8-DPU configs as the source of truth and
            # derive a 16-way feature tile at runtime.
            source = pimdl / "configs/host_replay_mha_8dpu.yaml"
            config = args.results_dir / "host_replay_mha_16dpu.yaml"
            if not config.exists():
                with source.open(encoding="utf-8") as stream:
                    cfg16 = yaml.safe_load(stream)
                cfg16["system_params"]["dpu_num"] = 16
                for name in PROJECTIONS:
                    cfg16["kernel_params"][f"{name}_lut_parallelism"] = 16
                    cfg16["kernel_params"][f"{name}_feature_mtile_size"] = 16
                config.write_text(yaml.safe_dump(cfg16, sort_keys=False), encoding="utf-8")
        else:
            config = (pimdl / (f"configs/cosim_results_{dpus}dpu.yaml" if args.model == "tiny"
                               else f"configs/host_replay_mha_{dpus}dpu.yaml"))
        cfg = configuration(config)
        result_dir = args.results_dir / f"{dpus}dpu"
        dump_dir = result_dir / "dumps"
        dump_dir.mkdir(parents=True, exist_ok=True)
        host_log = result_dir / "pimdl_host.log"
        if args.resume and host_log.exists():
            host_text = host_log.read_text(encoding="utf-8")
        else:
            print(f"[{dpus} DPU] PIM-DL host measurement", flush=True)
            host_text = build_pimdl(pimdl, config, dump_dir, env)
            host_log.write_text(host_text, encoding="utf-8")
        attention_ms, reorder_ms, other_ms = parse_host(host_text)
        build_dir = result_dir / "benchmark_build"
        if not (args.resume and build_dir.exists()):
            build_upim_kernels(upim, cfg, build_dir, args.upmem_home, env)
        print(f"[{dpus} DPU] uPIMulator kernels and transfers", flush=True)
        pim_ms, transfer_ms, run_projection_rows = run_upim(
            upim, cfg, dump_dir, result_dir, build_dir, env)
        projection_rows.extend(run_projection_rows)
        host_ms = attention_ms + reorder_ms + other_ms
        xfer_models = "+".join(sorted({row["xfer_model"] for row in run_projection_rows}))
        rows.append({"dpus": dpus, "host_ms": host_ms, "transfer_ms": transfer_ms,
                     "pim_ms": pim_ms, "attention_ms": attention_ms,
                     "reorder_ms": reorder_ms, "other_host_ms": other_ms,
                     "total_ms": host_ms + transfer_ms + pim_ms,
                     "xfer_model": xfer_models,
                     "validated_output": "all_dpus" if dpus == 1 else "dpu0_shard"})

    csv_path = args.results_dir / "results.csv"
    if args.dpus and csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            previous = list(csv.DictReader(stream))
        replaced = {str(row["dpus"]) for row in rows}
        retained = []
        for row in previous:
            if row["dpus"] not in replaced:
                retained.append({
                    key: (int(value) if key == "dpus" else
                          value if key in ("xfer_model", "validated_output") else
                          float(value))
                    for key, value in row.items()
                })
        rows = retained + rows
        rows.sort(key=lambda row: int(row["dpus"]))
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "dpus", "host_ms", "transfer_ms", "pim_ms", "attention_ms",
            "reorder_ms", "other_host_ms", "total_ms", "xfer_model",
            "validated_output"))
        writer.writeheader()
        writer.writerows(rows)

    projection_csv_path = args.results_dir / "projection_results.csv"
    if args.dpus and projection_csv_path.exists():
        with projection_csv_path.open(newline="", encoding="utf-8") as stream:
            previous_projection_rows = list(csv.DictReader(stream))
        replaced = {str(row["dpus"]) for row in projection_rows}
        projection_rows = [row for row in previous_projection_rows
                           if row["dpus"] not in replaced] + projection_rows
        projection_order = {name: index for index, name in enumerate(PROJECTIONS)}
        projection_rows.sort(
            key=lambda row: (int(row["dpus"]), projection_order[row["projection"]]))
    with projection_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=projection_rows[0].keys())
        writer.writeheader()
        writer.writerows(projection_rows)

    metadata = {
        "pim_aggregation": "sum projections; each projection uses maximum DPU logic cycles; kernel completion drains DPU-local memory",
        "transfer_aggregation": "sum projection HostTransfer wall cycles",
        "multi_dpu_validation": "timing launches every DPU; output bit-match currently covers DPU 0 shard",
        "warning": "Compare runs only when xfer_model matches",
    }
    (args.results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("\nDPUs  Host(ms)  Transfer(ms)  PIM(ms)  Attention(ms)  Reorder(ms)  Other host(ms)  Total(ms)")
    for row in rows:
        print(f"{row['dpus']:>4}  {row['host_ms']:>8.3f}  {row['transfer_ms']:>12.3f}  "
              f"{row['pim_ms']:>7.3f}  {row['attention_ms']:>13.3f}  "
              f"{row['reorder_ms']:>11.3f}  {row['other_host_ms']:>14.3f}  {row['total_ms']:>9.3f}")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {projection_csv_path}")


if __name__ == "__main__":
    main()
