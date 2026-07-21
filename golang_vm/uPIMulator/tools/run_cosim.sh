#!/usr/bin/env bash
#
# Orchestrate the PIM-DL <-> uPIMulator offline co-simulation for the four projection
# boundaries. For each projection:
#   1) stage PIM-DL's emitted LUT/index shard into a patch dir (cosim_prepare.py)
#   2) run the ported kernel on uPIMulator's cycle-accurate DPU+MRAM, injecting that
#      data via the mram-patch and dumping the DPU-tiled output (pimdl_output.bin)
#   3) bit-match the simulated output against PIM-DL's dumped reference (cosim_check.py)
#
# Prereq: run PIM-DL's emitter first to produce <DUMP_DIR>:
#   (in PIM-DL-ASPLOS/inference-engine)  python3 run_cosim_emit.py --dump_dir <DUMP_DIR>
#
# Usage:
#   tools/run_cosim.sh <DUMP_DIR> [projection ...]     # default: all four
#
# Env:
#   SKIP_COMPILE=1 (default)  reuse prebuilt DPU/SDK assembly (Docker-free);
#                             run tools/build_dpu_local.sh first.
#   SKIP_COMPILE=0            let uPIMulator compile via Docker/UPMEM.
#   COSIM_XFER_MODE=fixed_bw  (default) paper size/BW transfer model — safe for 16 DPUs.
#   COSIM_XFER_MODE=cycle     cycle-accurate SimulateMemory — ONLY safe with 1 DPU.
#   NUM_CHANNELS / NUM_RANKS_PER_CHANNEL / NUM_DPUS_PER_RANK  topology (default 1×2×8=16).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="${ROOT}/tools"
DUMP_DIR="${1:?usage: run_cosim.sh <DUMP_DIR> [projection ...]}"; shift || true
DUMP_DIR="$(cd "${DUMP_DIR}" && pwd)"
PROJECTIONS=("$@"); [[ ${#PROJECTIONS[@]} -eq 0 ]] && PROJECTIONS=(qkv o ffn1 ffn2)

SKIP_COMPILE="${SKIP_COMPILE:-1}"
# Match cosim_static_small.yaml dpu_num=16 (2 ranks × 8 DPUs/rank, UPMEM-like).
NUM_CHANNELS="${NUM_CHANNELS:-1}"
NUM_RANKS_PER_CHANNEL="${NUM_RANKS_PER_CHANNEL:-2}"
NUM_DPUS_PER_RANK="${NUM_DPUS_PER_RANK:-8}"
# fixed_bw is mandatory for multi-DPU; cycle-accurate xfer OOMs at 16 DPUs.
COSIM_XFER_MODE="${COSIM_XFER_MODE:-fixed_bw}"
export COSIM_XFER_MODE
BIN_DIR="${ROOT}/bin_cosim"
LOG_DIR="${ROOT}/cosim_logs"
UPIM="${ROOT}/build/uPIMulator"
mkdir -p "${BIN_DIR}"
rm -rf "${LOG_DIR}" && mkdir -p "${LOG_DIR}"
BENCH_BUILD_DIR="${BENCH_BUILD_DIR:-${ROOT}/benchmark/build_user}"
export UPIM_BENCH_BUILD_DIR="${UPIM_BENCH_BUILD_DIR:-${BENCH_BUILD_DIR}}"

TOTAL_DPUS=$((NUM_CHANNELS * NUM_RANKS_PER_CHANNEL * NUM_DPUS_PER_RANK))
echo "co-sim topology: ${NUM_CHANNELS} ch × ${NUM_RANKS_PER_CHANNEL} rank × ${NUM_DPUS_PER_RANK} dpu = ${TOTAL_DPUS} DPUs"
echo "co-sim xfer mode: ${COSIM_XFER_MODE}"
if [[ "${COSIM_XFER_MODE}" == "cycle" && "${TOTAL_DPUS}" -gt 1 ]]; then
    echo "WARNING: COSIM_XFER_MODE=cycle with ${TOTAL_DPUS} DPUs can crash the machine; forcing fixed_bw."
    COSIM_XFER_MODE=fixed_bw
    export COSIM_XFER_MODE
fi

[[ -x "${UPIM}" ]] || { echo "building simulator..."; (cd "${ROOT}" && go build -o build/uPIMulator ./src); }

now_ms() {
    date +%s%3N
}

pass=0; fail=0
for proj in "${PROJECTIONS[@]}"; do
    echo "================ ${proj} ================"
    PATCH_DIR="${ROOT}/cosim_patch/${proj}"
    rm -rf "${PATCH_DIR}"; mkdir -p "${PATCH_DIR}"
    t_proj_start="$(now_ms)"

    # 1) stage data; capture BENCHMARK + PIMDL_OUTPUT_BYTES from stdout
    t0="$(now_ms)"
    prep_out="$(python3 "${TOOLS}/cosim_prepare.py" "${proj}" --dump_dir "${DUMP_DIR}" --patch_dir "${PATCH_DIR}")"
    t1="$(now_ms)"
    eval "${prep_out}"   # sets BENCHMARK, PIMDL_OUTPUT_BYTES

    # 2) run the simulator (bin dir is wiped by the sim; patch dir is separate)
    t2="$(now_ms)"
    PIMDL_OUTPUT_BYTES="${PIMDL_OUTPUT_BYTES}" "${UPIM}" \
        --benchmark "${BENCHMARK}" \
        --root_dirpath "${ROOT}" \
        --bin_dirpath "${BIN_DIR}" \
        --pimdl_patch_dirpath "${PATCH_DIR}" \
        --num_channels "${NUM_CHANNELS}" \
        --num_ranks_per_channel "${NUM_RANKS_PER_CHANNEL}" \
        --num_dpus_per_rank "${NUM_DPUS_PER_RANK}" \
        --num_tasklets 16 --skip_compile "${SKIP_COMPILE}" --verbose 0
    t3="$(now_ms)"

    # 3) bit-match
    t4="$(now_ms)"
    if python3 "${TOOLS}/cosim_check.py" "${proj}" --bin_dir "${BIN_DIR}" --dump_dir "${DUMP_DIR}"; then
        pass=$((pass+1))
    else
        fail=$((fail+1))
    fi
    t5="$(now_ms)"
    # Persist per-projection logs for later stats aggregation.
    cp "${BIN_DIR}/log.txt" "${LOG_DIR}/log_${proj}.txt"
    {
        echo "projection=${proj}"
        echo "prepare_ms=$((t1 - t0))"
        echo "simulate_ms=$((t3 - t2))"
        echo "check_ms=$((t5 - t4))"
        echo "host_total_ms=$((t5 - t_proj_start))"
    } > "${LOG_DIR}/timing_${proj}.txt"
    echo "  DPU cycles (log.txt):"
    grep -E "Logic\[.*\]_logic_cycle|MemoryController\[.*\]_memory_cycle" "${BIN_DIR}/log.txt" 2>/dev/null | sed 's/^/    /' || true
    echo "  Host↔MRAM transfer cycles (log.txt):"
    grep -E "HostTransfer_" "${BIN_DIR}/log.txt" 2>/dev/null | sed 's/^/    /' || true
done

echo "======================================="
echo "co-simulation bit-match: ${pass} PASS, ${fail} FAIL"
[[ ${fail} -eq 0 ]]
