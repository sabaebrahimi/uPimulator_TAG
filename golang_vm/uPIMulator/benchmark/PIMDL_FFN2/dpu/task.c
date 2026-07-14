/*
 * FFN2 projection boundary. Shares the one ported LUT kernel; only the tile-size
 * -D macros differ (see dpu/CMakeLists.txt). FFN2 has the widest codebook count.
 */
#include "../../PIMDL/dpu/task.c"
