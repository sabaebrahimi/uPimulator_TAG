# PIMDL = QKV projection boundary (co-simulation, Phase 1).
# Tile sizes are a downscaled-but-representative STATIC config that fits the
# 128 KB DPU WRAM (real BERT-base QKV static LUT is 2.3 MB). NUM_CENTROID and the
# n_stile/n_mtile geometry are kept at the real sim_small values; NUM_CODEBOOK and
# FEATURE_STILE_SIZE are downscaled. This config MUST match the PIM-DL emitter's
# cosim_static_small.yaml QKV entry byte-for-byte for the bit-match to hold.
SET(N_STILE_SIZE       64)
SET(FEATURE_STILE_SIZE 48)
SET(N_MTILE_SIZE       64)
SET(FEATURE_MTILE_SIZE 24)
SET(CB_MTILE_SIZE      16)
SET(NUM_CODEBOOK       32)
SET(NUM_CENTROID       16)

set(CMAKE_C_COMPILER "/root/upmem-2023.2.0-Linux-x86_64/bin/dpu-upmem-dpurte-clang")
set(CMAKE_C_FLAGS "-w -I/root/uPIMulator/benchmark/PIMDL/support -O2 -S -DNR_TASKLETS=${NR_TASKLETS} \
-DN_STILE_SIZE=${N_STILE_SIZE} -DFEATURE_STILE_SIZE=${FEATURE_STILE_SIZE} \
-DN_MTILE_SIZE=${N_MTILE_SIZE} -DFEATURE_MTILE_SIZE=${FEATURE_MTILE_SIZE} \
-DCB_MTILE_SIZE=${CB_MTILE_SIZE} -DNUM_CODEBOOK=${NUM_CODEBOOK} -DNUM_CENTROID=${NUM_CENTROID}")

file(GLOB_RECURSE SRCS *.c)

add_executable(PIMDL_device ${SRCS})
