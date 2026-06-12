#include <stdlib.h>
#include <dpu.h>
#define NUM_DPUS 1
#define NUM_TASKLETS 1
#define DATA_PREP_PARAMS 64

struct dpu_arguments_t {
	uint32_t input_height;
};

int main(void) {
	struct dpu_set_t dpu_set;
	struct dpu_set_t dpu;
	int nr_of_dpus = NUM_DPUS;

	dpu_alloc(nr_of_dpus, NULL, &dpu_set);
	dpu_load(dpu_set, DPU_BINARY, NULL);

	struct dpu_arguments_t args;
	args.input_height = (uint32_t)DATA_PREP_PARAMS;

	int i = 0;
	DPU_FOREACH(dpu_set, dpu, i) {
		dpu_prepare_xfer(dpu, &args);
	}
	dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0,
		      sizeof(struct dpu_arguments_t), DPU_XFER_DEFAULT);

	dpu_launch(dpu_set, DPU_SYNCHRONOUS);

	dpu_free(dpu_set);
	return 0;
}
