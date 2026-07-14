/*
 * O projection boundary. Shares the one ported LUT kernel; only the tile-size
 * -D macros differ (see dpu/CMakeLists.txt). The kernel's own relative include
 * ("../support/common.h") resolves against benchmark/PIMDL/dpu, so no local
 * support copy is needed.
 */
#include "../../PIMDL/dpu/task.c"
