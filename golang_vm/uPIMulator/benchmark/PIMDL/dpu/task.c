/*
 * Minimal PIM-DL-style placeholder kernel for uPIMulator integration.
 * Real LUT kernels from PIM-DL can replace this file once the toolchain path is stable.
 *
 * MRAM symbols below are patch targets for program.RunPimdlMramPatch (see pimdl_mram_patch.go).
 */
#include <stdint.h>
#include <defs.h>
#include <mram.h>
#include <barrier.h>
#include <alloc.h>

#include "../support/common.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

__mram_noinit uint8_t lut_table[256];
__mram_noinit uint16_t input_index[128];
__mram_noinit int32_t output_data[32];

BARRIER_INIT(pimdl_barrier, NR_TASKLETS);

int main(void) {
	unsigned int tasklet_id = me();

	if (tasklet_id == 0) {
		mem_reset();
	}
	barrier_wait(&pimdl_barrier);

	(void)DPU_INPUT_ARGUMENTS.input_height;

	if (tasklet_id == 0) {
		int32_t acc = 0;
		for (int i = 0; i < 32; i++) {
			uint8_t lut = lut_table[(unsigned)i];
			uint16_t idx = input_index[(unsigned)i];
			acc += (int32_t)lut + (int32_t)idx;
			output_data[i] = acc;
		}
	}

	return 0;
}
