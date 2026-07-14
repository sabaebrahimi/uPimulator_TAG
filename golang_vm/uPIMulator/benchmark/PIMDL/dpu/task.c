/*
 * PIM-DL LUT kernel ported into uPIMulator (co-simulation, Phase 1).
 *
 * This is the real PIM-DL DPU kernel, restricted to the single variant the TAG
 * project targets: STATIC_LUT_TABLE + LOOP_ORDER_NFC.
 * The kernel body (lut_kernel) is kept byte-for-byte identical to
 *   PIM-DL-ASPLOS/inference-engine/src/dpu/pim_lut_kernel.c  (lines 97-204)
 * so that uPIMulator's cycle-accurate DPU reproduces PIM-DL's DPU compute
 * bit-for-bit for a given (LUT table, input_index) snapshot.
 *
 * Tile-size macros are injected at compile time via -D flags (see dpu/CMakeLists.txt),
 * exactly as PIM-DL does through run_layer.py:
 *   N_STILE_SIZE, FEATURE_STILE_SIZE, N_MTILE_SIZE, FEATURE_MTILE_SIZE,
 *   CB_MTILE_SIZE, NUM_CODEBOOK, NUM_CENTROID
 * NR_TASKLETS is injected by the benchmark build (build.py / CMake).
 *
 * MRAM symbols lut_table / input_index / output_data are the patch + readback
 * targets used by program.RunPimdlMramPatch (pimdl_mram_patch.go) and the host
 * readback (host/app.c).
 *
 * The FINE_GRAIN / COARSE_GRAIN load paths and the non-NFC loop orders are
 * intentionally omitted; the STATIC+NFC path never references the *_LOAD_TILE_SIZE
 * macros, so this file needs no load-tile defines.
 */
#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <barrier.h>

#include "../support/common.h"


/* ---- derived tile geometry (subset used by STATIC+NFC) ---- */
#define INPUT_STILE_SIZE (N_STILE_SIZE * CB_MTILE_SIZE)
#define OUTPUT_STILE_SIZE (N_STILE_SIZE * FEATURE_MTILE_SIZE)
#define INPUT_MTILE_SIZE (N_MTILE_SIZE * CB_MTILE_SIZE)
#define OUTPUT_MTILE_SIZE (N_MTILE_SIZE * FEATURE_MTILE_SIZE)
#define LUT_CBTILE_SIZE (NUM_CENTROID * FEATURE_STILE_SIZE)
#define LUT_CBMTILE_SIZE (CB_MTILE_SIZE * NUM_CENTROID * FEATURE_STILE_SIZE)
#define INPUT_BUFFER_SIZE_PER_TASKLET (INPUT_MTILE_SIZE / NR_TASKLETS)
#define OUTPUT_BUFFER_SIZE_PER_TASKLET (OUTPUT_MTILE_SIZE / NR_TASKLETS)
#define LUT_BUFFER_SIZE_PER_TASKLET (FEATURE_STILE_SIZE * NUM_CODEBOOK * NUM_CENTROID / NR_TASKLETS)
#define INPUT_BUFFER_BYTE_PER_TASKLET  (INPUT_BUFFER_SIZE_PER_TASKLET * INDEX_SIZE)
#define OUTPUT_BUFFER_BYTE_PER_TASKLET (OUTPUT_BUFFER_SIZE_PER_TASKLET * OUTPUT_SIZE)
#define LUT_BUFFER_BYTE_PER_TASKLET (LUT_BUFFER_SIZE_PER_TASKLET * LUT_SIZE)
#define MAX_INPUT_PER_RW (2048 / INDEX_SIZE)
#define MAX_OUTPUT_PER_RW (2048 / OUTPUT_SIZE)
#define MAX_LUT_PER_RW (2048 / LUT_SIZE)
#define N_PER_TASKLET (N_MTILE_SIZE / NR_TASKLETS)


/*------------------LUT table-----------------------*/
__mram_noinit lut_data_type lut_table[FEATURE_STILE_SIZE * NUM_CODEBOOK * NUM_CENTROID];

/*------------------Index data-----------------------*/
__mram_noinit index_data_type input_index[N_STILE_SIZE * NUM_CODEBOOK];

/*------------------Output data-----------------------*/
__mram_noinit output_data_type output_data[N_STILE_SIZE * FEATURE_STILE_SIZE];

/*------------------Operation Parameters-----------------------*/
__host dpu_arguments_t dpu_arguments;


/*------------------WRAM Buffers (shared by all tasklets)-----------------------*/
__dma_aligned index_data_type input_index_buffer[INPUT_MTILE_SIZE];
__dma_aligned output_data_type output_result_buffer[OUTPUT_MTILE_SIZE];
__dma_aligned lut_data_type lut_table_buffer[FEATURE_STILE_SIZE * NUM_CODEBOOK * NUM_CENTROID];


/*------------------Barrier for tasklet sync-----------------------*/
BARRIER_INIT(lut_load_barrier, NR_TASKLETS);


void lut_kernel();


int main()
{
    lut_kernel();
    return 0;
}


void lut_kernel()
{
    // tmp taklet id
    uint32_t tasklet_id = me();

    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;

    // load lut table first
    uint32_t lut_tensor_offset = lut_buffer_offset;
#if LUT_BUFFER_BYTE_PER_TASKLET > 2048
    uint32_t tmp_lut_offset = 0;
    uint32_t tmp_lut_byte_offset = 0;
    for(tmp_lut_byte_offset=0; tmp_lut_byte_offset<LUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_lut_byte_offset+=2048)
    {
        mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], 2048);
        tmp_lut_offset += MAX_LUT_PER_RW;
    }
    mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], LUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
    mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);
#endif
    barrier_wait(&lut_load_barrier);

    // tile offset, changing along with loop iteration
    uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
    uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE
    for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
    {
        uint32_t output_stile_offset = 0;
        for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
        {
            // reset output buffer
            // when oc is the innermost outer iterator, we can just reset output buffer instead of loading output data
            for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                output_result_buffer[tmp_output_offset] = 0;

            uint32_t input_stile_offset = 0;
            uint32_t lut_cbmtile_offset = 0;
            for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
            {
                uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
                // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_input_offset = 0;
                uint32_t tmp_input_byte_offset = 0;
                for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
                {
                    mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                    tmp_input_offset += MAX_INPUT_PER_RW;
                }
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
                mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // read lut and compute
                uint32_t tmp_input_row_offset = 0;
                uint32_t tmp_output_row_offset = 0;
                for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                {
                    uint32_t lut_cbtile_offset = 0;
                    for(uint32_t tmp_cb=0; tmp_cb<CB_MTILE_SIZE; ++tmp_cb)
                    {
                        uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_cb];
                        for(uint32_t tmp_f=0; tmp_f<FEATURE_MTILE_SIZE; ++tmp_f)
                        {
                            output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_f] += lut_table_buffer[lut_cbmtile_offset + lut_cbtile_offset + tmp_index + tmp_init_feature + tmp_f];
                        }

                        lut_cbtile_offset += LUT_CBTILE_SIZE;
                    }

                    tmp_input_row_offset += CB_MTILE_SIZE;
                    tmp_output_row_offset += FEATURE_MTILE_SIZE;
                }

                // update input's stile offset
                input_stile_offset += INPUT_STILE_SIZE;
                lut_cbmtile_offset += LUT_CBMTILE_SIZE;
            }

            // save output buffer to DRAM
            uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
            uint32_t tmp_output_offset = 0;
            uint32_t tmp_output_byte_offset = 0;
            for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
            {
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                tmp_output_offset += MAX_OUTPUT_PER_RW;
            }
            mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
            mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif

            // update output's stile offset
            output_stile_offset += OUTPUT_STILE_SIZE;
        }

        // update input & output intra each stile's mtile offset
        input_mtile_offset += INPUT_MTILE_SIZE;
        output_mtile_offset += OUTPUT_MTILE_SIZE;
    }
}
