# PIMDL_O = O projection boundary. See benchmark/PIMDL/dpu/CMakeLists.txt for the
# config rationale. Must match cosim_static_small.yaml's O entry byte-for-byte.
SET(N_STILE_SIZE       64)
SET(FEATURE_STILE_SIZE 16)
SET(N_MTILE_SIZE       64)
SET(FEATURE_MTILE_SIZE 16)
SET(CB_MTILE_SIZE      16)
SET(NUM_CODEBOOK       32)
SET(NUM_CENTROID       16)

set(CMAKE_C_COMPILER "/root/upmem-2023.2.0-Linux-x86_64/bin/dpu-upmem-dpurte-clang")
set(CMAKE_C_FLAGS "-w -I/root/uPIMulator/benchmark/PIMDL/support -O2 -S -DNR_TASKLETS=${NR_TASKLETS} \
-DN_STILE_SIZE=${N_STILE_SIZE} -DFEATURE_STILE_SIZE=${FEATURE_STILE_SIZE} \
-DN_MTILE_SIZE=${N_MTILE_SIZE} -DFEATURE_MTILE_SIZE=${FEATURE_MTILE_SIZE} \
-DCB_MTILE_SIZE=${CB_MTILE_SIZE} -DNUM_CODEBOOK=${NUM_CODEBOOK} -DNUM_CENTROID=${NUM_CENTROID}")

file(GLOB_RECURSE SRCS *.c)

add_executable(PIMDL_O_device ${SRCS})
