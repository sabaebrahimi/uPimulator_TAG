extern "C"{
    #include <dpu.h>
}
#include <omp.h>
#include <vector>
#include <iostream>

#include "pim_lut_host.h"
#include "utils.h"
#include "dpu_common.h"

#ifdef PIMDL_COSIM_DUMP
#include <cstdio>
#include <cstdlib>
// Co-simulation emitter (uPIMulator TAG project).
// Dumps the exact per-DPU bytes exchanged with the DPUs at a projection boundary so
// uPIMulator can replay the same kernel on its cycle-accurate DPU+MRAM and bit-match.
// Files are tagged by the (num_codebook, feature_stile_size) signature, which is
// unique per projection in the co-sim config, so dumps are order-independent.
// Only DPU 0's shard is dumped (the clean 1:1 unit for the bit-match).
static void pimdl_cosim_dump(const char* kind, uint32_t num_codebook, uint32_t feature_stile_size,
                             const void* data, size_t nbytes)
{
    const char* dir = getenv("PIMDL_COSIM_DUMP_DIR");
    if(dir == nullptr) dir = ".";
    const char* tag = getenv("PIMDL_COSIM_DUMP_TAG");
    char path[1024];
    if(tag != nullptr && tag[0] != '\0')
        snprintf(path, sizeof(path), "%s/%s_%s_cb%u_fs%u.bin", dir, kind, tag, num_codebook, feature_stile_size);
    else
        snprintf(path, sizeof(path), "%s/%s_cb%u_fs%u.bin", dir, kind, num_codebook, feature_stile_size);
    FILE* f = fopen(path, "wb");
    if(f == nullptr) { fprintf(stderr, "pimdl_cosim_dump: cannot open %s\n", path); return; }
    fwrite(data, 1, nbytes, f);
    fclose(f);
    fprintf(stderr, "pimdl_cosim_dump: wrote %zu bytes to %s\n", nbytes, path);
}
#endif


void prepare_parameters(dpu_set_t* dpu_set, dpu_arguments_t* dpu_arguments)
{
    uint32_t each_dpu;
    dpu_set_t dpu;
    DPU_FOREACH(*dpu_set, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, dpu_arguments));
    }
    DPU_ASSERT(dpu_push_xfer(*dpu_set, DPU_XFER_TO_DPU, "dpu_arguments", 0, sizeof(dpu_arguments_t), DPU_XFER_ASYNC));
    dpu_sync(*dpu_set);
}


void prepare_lut_table(LUTParams lut_params, dpu_set_t* dpu_set, lut_data_type* lut_table)
{
    uint32_t table_offset = lut_params.feature_stile_size * lut_params.num_centroid * lut_params.num_codebook;
    uint32_t each_dpu;
    dpu_set_t dpu;
    DPU_FOREACH(*dpu_set, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &lut_table[table_offset * (each_dpu % lut_params.lut_parallelism)]));
    }
    DPU_ASSERT(dpu_push_xfer(*dpu_set, DPU_XFER_TO_DPU, "lut_table", 0, sizeof(lut_data_type)*table_offset, DPU_XFER_ASYNC));
    dpu_sync(*dpu_set);
}


void prepare_input_index(LUTParams lut_params, dpu_set_t* dpu_set, index_data_type* input_index)
{
    if(lut_params.lut_load_type == STATIC)
    {
        #pragma omp parallel for num_threads(lut_params.num_threads)
        for(uint32_t i=0; i<lut_params.n; ++i)
        {
            uint32_t tmp_input_group = i / lut_params.n_stile_size;
            uint32_t tmp_intra_group_id = i % lut_params.n_stile_size;
            for(uint32_t tmp_codebook=0; tmp_codebook<lut_params.num_codebook; ++tmp_codebook)
            {
                uint32_t tmp_cb_tile = tmp_codebook / lut_params.cb_mtile_size;
                uint32_t tmp_intra_tile_id = tmp_codebook % lut_params.cb_mtile_size;

                uint32_t tmp_offset = tmp_input_group * lut_params.n_stile_size * lut_params.num_codebook
                                    + tmp_cb_tile * lut_params.n_stile_size * lut_params.cb_mtile_size
                                    + tmp_intra_group_id * lut_params.cb_mtile_size
                                    + tmp_intra_tile_id;

                input_index[tmp_offset] = input_index[tmp_offset] * lut_params.feature_stile_size;
            }
        }
    }
    else if(lut_params.lut_load_type == FINE_GRAIN)
    {
        #pragma omp parallel for num_threads(lut_params.num_threads)
        for(uint32_t i=0; i<lut_params.n; ++i)
        {
            uint32_t tmp_input_group = i / lut_params.n_stile_size;
            uint32_t tmp_intra_group_id = i % lut_params.n_stile_size;
            for(uint32_t tmp_codebook=0; tmp_codebook<lut_params.num_codebook; ++tmp_codebook)
            {
                uint32_t tmp_cb_tile = tmp_codebook / lut_params.cb_mtile_size;
                uint32_t tmp_intra_tile_id = tmp_codebook % lut_params.cb_mtile_size;

                uint32_t tmp_offset = tmp_input_group * lut_params.n_stile_size * lut_params.num_codebook
                                    + tmp_cb_tile * lut_params.n_stile_size * lut_params.cb_mtile_size
                                    + tmp_intra_group_id * lut_params.cb_mtile_size
                                    + tmp_intra_tile_id;

                input_index[tmp_offset] = input_index[tmp_offset] * lut_params.feature_load_tile_size;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(lut_params.num_threads)
        for(uint32_t i=0; i<lut_params.n; ++i)
        {
            uint32_t tmp_input_group = i / lut_params.n_stile_size;
            uint32_t tmp_intra_group_id = i % lut_params.n_stile_size;
            for(uint32_t tmp_codebook=0; tmp_codebook<lut_params.num_codebook; ++tmp_codebook)
            {
                uint32_t tmp_cb_tile = tmp_codebook / lut_params.cb_mtile_size;
                uint32_t tmp_intra_tile_id = tmp_codebook % lut_params.cb_mtile_size;

                uint32_t tmp_intra_load_tile_id = tmp_intra_tile_id % lut_params.cb_load_tile_size;

                uint32_t tmp_offset = tmp_input_group * lut_params.n_stile_size * lut_params.num_codebook
                                    + tmp_cb_tile * lut_params.n_stile_size * lut_params.cb_mtile_size
                                    + tmp_intra_group_id * lut_params.cb_mtile_size
                                    + tmp_intra_tile_id;

                input_index[tmp_offset] = tmp_intra_load_tile_id * lut_params.num_centroid * lut_params.feature_load_tile_size
                                        + input_index[tmp_offset] * lut_params.feature_load_tile_size;
            }
        }
    }

    uint32_t input_offset = lut_params.n_stile_size * lut_params.num_codebook;
    uint32_t each_dpu;
    dpu_set_t dpu;
    DPU_FOREACH(*dpu_set, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &input_index[input_offset * (uint32_t)(each_dpu / lut_params.lut_parallelism)]));
    }
    DPU_ASSERT(dpu_push_xfer(*dpu_set, DPU_XFER_TO_DPU, "input_index", 0, sizeof(index_data_type)*input_offset, DPU_XFER_ASYNC));
    dpu_sync(*dpu_set);
}


void pim_lut(LUTParams lut_params, dpu_set_t* dpu_set,
             index_data_type* input_index, lut_data_type* lut_table, float* bias_tensor, float* output_tensor)
{
#ifdef AMM_BREAKDOWN
    // time variables
    double time1, time2;
#endif

#ifdef AMM_BREAKDOWN
    time1 = W_time();
#endif
    // buffer to hold output lut data
    output_data_type* output_lut_data = new output_data_type[sizeof(output_data_type) * lut_params.n * lut_params.output_feature_len];
#ifdef AMM_BREAKDOWN
    time2 = W_time();
    amm_profiles.other_latency += time2 - time1;
#endif

#ifdef AMM_BREAKDOWN
    time1 = W_time();
#endif
    // load lut table into DPUs
    prepare_lut_table(lut_params, dpu_set, lut_table);
#ifdef PIMDL_COSIM_DUMP
    // DPU 0's LUT shard == lut_table[0 .. table_offset]; matches uPIMulator
    // lut_table[FEATURE_STILE_SIZE*NUM_CODEBOOK*NUM_CENTROID] (int8).
    pimdl_cosim_dump("lut", lut_params.num_codebook, lut_params.feature_stile_size,
        lut_table,
        (size_t)lut_params.feature_stile_size * lut_params.num_centroid * lut_params.num_codebook * sizeof(lut_data_type));
#endif
#ifdef AMM_BREAKDOWN
    time2 = W_time();
    amm_profiles.data_transfer_latency += time2 - time1;
#endif

#ifdef AMM_BREAKDOWN
    time1 = W_time();
#endif
    // load inputs into DPUs
    prepare_input_index(lut_params, dpu_set, input_index);
#ifdef PIMDL_COSIM_DUMP
    // DPU 0's index shard == input_index[0 .. input_offset], AFTER the STATIC
    // pre-multiply (index *= feature_stile_size); matches uPIMulator
    // input_index[N_STILE_SIZE*NUM_CODEBOOK] (uint16).
    pimdl_cosim_dump("index", lut_params.num_codebook, lut_params.feature_stile_size,
        input_index,
        (size_t)lut_params.n_stile_size * lut_params.num_codebook * sizeof(index_data_type));
#endif
#ifdef AMM_BREAKDOWN
    time2 = W_time();
    amm_profiles.data_transfer_latency += time2 - time1;
#endif

#ifdef AMM_BREAKDOWN
    time1 = W_time();
#endif
    // launch kernel
    DPU_ASSERT(dpu_launch(*dpu_set, DPU_SYNCHRONOUS));
#ifdef AMM_BREAKDOWN
    time2 = W_time();
    amm_profiles.kernel_latency += time2 - time1;
#endif

#ifdef AMM_BREAKDOWN
    time1 = W_time();
#endif
    // read output from DPUs
    uint32_t output_offset = lut_params.n_stile_size * lut_params.feature_stile_size;
    uint32_t each_dpu;
    dpu_set_t dpu;
    DPU_FOREACH(*dpu_set, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &output_lut_data[output_offset * each_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(*dpu_set, DPU_XFER_FROM_DPU, "output_data", 0, sizeof(output_data_type) * output_offset, DPU_XFER_ASYNC));
    dpu_sync(*dpu_set);
#ifdef PIMDL_COSIM_DUMP
    // DPU 0's DPU-tiled output BEFORE rescale/reorder == output_lut_data[0 .. output_offset];
    // this is the golden reference uPIMulator's simulated output_data must bit-match.
    pimdl_cosim_dump("output", lut_params.num_codebook, lut_params.feature_stile_size,
        output_lut_data,
        (size_t)lut_params.n_stile_size * lut_params.feature_stile_size * sizeof(output_data_type));
#endif
#ifdef AMM_BREAKDOWN
    time2 = W_time();
    amm_profiles.data_transfer_latency += time2 - time1;
#endif

#ifdef AMM_BREAKDOWN
    time1 = W_time();
#endif
    // scale output to float values
    uint32_t mtile_offset = lut_params.n_stile_size * lut_params.feature_mtile_size;
    #pragma omp parallel for num_threads(lut_params.num_threads)
    for(int dpu_id=0; dpu_id<lut_params.dpu_num; ++dpu_id)
    {
        int output_dpu_id = dpu_id * output_offset;
        int bias_dpu_id = (dpu_id % lut_params.lut_parallelism) * lut_params.feature_stile_size;
        for(int i=0; i<output_offset; ++i)
        {
            int bias_init_id = (i / mtile_offset) * lut_params.feature_mtile_size;
            int bias_intra_id = (i % lut_params.feature_mtile_size);
            output_tensor[output_dpu_id+i] = output_lut_data[output_dpu_id+i]*lut_params.scale + lut_params.bias*lut_params.num_codebook + bias_tensor[bias_dpu_id+bias_init_id+bias_intra_id];
        }
    }    

#ifdef AMM_BREAKDOWN
    time2 = W_time();
    amm_profiles.other_latency += time2 - time1;
#endif

#ifdef AMM_BREAKDOWN
    printf("other latency %.6f, data transfer latency %.6f, pim kernel latency %.6f\n", amm_profiles.other_latency, amm_profiles.data_transfer_latency, amm_profiles.kernel_latency);
#endif

#ifdef DEBUG
    DPU_FOREACH(*dpu_set, dpu, each_dpu) 
    {
        if(each_dpu==0)
            DPU_ASSERT(dpu_log_read(dpu, stdout));
    }
#endif

}

