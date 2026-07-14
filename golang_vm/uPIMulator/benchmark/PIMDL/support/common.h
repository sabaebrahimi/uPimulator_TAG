#ifndef PIMDL_COMMON_H
#define PIMDL_COMMON_H

#include <stdint.h>

/*
 * PIM-DL data types and element sizes.
 * Mirrors PIM-DL-ASPLOS/inference-engine/src/dpu/dpu_configs.h so the ported
 * kernel and the co-simulation data snapshots agree byte-for-byte.
 */
typedef int8_t   lut_data_type;    /* LUT table entries    */
typedef uint16_t index_data_type;  /* per-codebook indices */
typedef int32_t  output_data_type; /* accumulated outputs  */

/* Element sizes in bytes (must match the typedefs above). */
#define INDEX_SIZE  2
#define LUT_SIZE    1
#define OUTPUT_SIZE 4

typedef struct {
    uint32_t input_height;
} dpu_arguments_t;

#endif
