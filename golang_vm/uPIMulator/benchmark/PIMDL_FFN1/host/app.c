#include <stdlib.h>
#include <dpu.h>

#define NUM_DPUS 1
#define NUM_TASKLETS 16

struct dpu_arguments_t {
    int input_height;
};

int main() {
    struct dpu_set_t dpu_set;
    struct dpu_set_t dpu;
    int nr_of_dpus = NUM_DPUS;
    struct dpu_arguments_t input_arguments;

    dpu_alloc(nr_of_dpus, NULL, &dpu_set);
    dpu_load(dpu_set, DPU_BINARY, NULL);

    input_arguments.input_height = 0;

    DPU_FOREACH(dpu_set, dpu, i)
    {
        dpu_prepare_xfer(dpu, &input_arguments);
    }
    dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "dpu_arguments", 0,
                  sizeof(struct dpu_arguments_t), DPU_XFER_DEFAULT);

    dpu_launch(dpu_set, DPU_SYNCHRONOUS);

    dpu_free(dpu_set);
    return 0;
}
