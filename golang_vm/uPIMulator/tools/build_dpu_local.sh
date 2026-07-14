#!/usr/bin/env bash
#
# Docker-free build of the four PIM-DL co-simulation DPU kernels using a LOCAL UPMEM
# SDK, placing each assembly where the uPIMulator linker expects it:
#   benchmark/build/<NAME>/dpu/CMakeFiles/<NAME>_device.dir/task.c.o   (ASCII assembly)
#
# Pair with `uPIMulator --skip_compile 1` so the simulator reuses these artifacts
# (and the already-built sdk/build assembly) instead of invoking Docker.
#
# The per-projection tile macros below MIRROR benchmark/<NAME>/dpu/CMakeLists.txt and
# configs/cosim_static_small.yaml. If you change one, change all three.
set -euo pipefail

UPMEM_HOME="${UPMEM_HOME:-/home/saba/master/upmem-2025.1.0-Linux-x86_64}"
CLANG="${UPMEM_HOME}/bin/dpu-upmem-dpurte-clang"
NR_TASKLETS="${NR_TASKLETS:-16}"

# uPIMulator root = parent of this tools/ dir.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPPORT="${ROOT}/benchmark/PIMDL/support"
BENCH_BUILD_DIR="${BENCH_BUILD_DIR:-${ROOT}/benchmark/build_user}"

if [[ ! -x "${CLANG}" ]]; then
    echo "error: dpu clang not found at ${CLANG}; set UPMEM_HOME" >&2
    exit 1
fi

# name : "N_STILE FEATURE_STILE N_MTILE FEATURE_MTILE CB_MTILE NUM_CODEBOOK NUM_CENTROID"
# LUT tile = FEATURE_STILE*NUM_CODEBOOK*NUM_CENTROID must fit real 64 KB DPU WRAM
# alongside 16 tasklet stacks (<= ~32 KB LUT; these are all <= 24 KB).
declare -A CFG=(
  [PIMDL]="64 48 64 24 16 32 16"
  [PIMDL_O]="64 16 64 16 16 32 16"
  [PIMDL_FFN1]="64 32 64 16 16 32 16"
  [PIMDL_FFN2]="64 16 64 16 16 64 16"
)

for NAME in PIMDL PIMDL_O PIMDL_FFN1 PIMDL_FFN2; do
    read -r NS FS NM FM CBM CB CT <<< "${CFG[$NAME]}"
    OUTDIR="${BENCH_BUILD_DIR}/${NAME}/dpu/CMakeFiles/${NAME}_device.dir"
    mkdir -p "${OUTDIR}"
    echo "[build] ${NAME}: N_STILE=${NS} FEATURE_STILE=${FS} CB=${CB} -> ${OUTDIR}/task.c.o"
    "${CLANG}" -w -I "${SUPPORT}" -O2 -S \
        -DNR_TASKLETS="${NR_TASKLETS}" \
        -DN_STILE_SIZE="${NS}" -DFEATURE_STILE_SIZE="${FS}" \
        -DN_MTILE_SIZE="${NM}" -DFEATURE_MTILE_SIZE="${FM}" \
        -DCB_MTILE_SIZE="${CBM}" -DNUM_CODEBOOK="${CB}" -DNUM_CENTROID="${CT}" \
        "${ROOT}/benchmark/${NAME}/dpu/task.c" -o "${OUTDIR}/task.c.o"
done

echo "[build] done. Run the simulator with --skip_compile 1."
