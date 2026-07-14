#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <cassert>
#include <limits>
#include <omp.h>
#include <memory.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <cstdlib>
#include <vector>

#include "ggml.h"

#include "parser.h"
#include "utils.h"
#include "defines.h"
#include "dpu_common.h"
#include "amm_host.h"
#include "transformer_layer.h"
#include "transformer_model.h"
#include"cnpy.h"
extern "C"{
    #include <dpu.h>
}

using namespace std;

static size_t get_env_size_mb_or_default(const char *env_name, size_t default_mb)
{
    const char *v = std::getenv(env_name);
    if (v == nullptr || *v == '\0')
        return default_mb * 1024ull * 1024ull;
    char *endptr = nullptr;
    unsigned long long parsed = std::strtoull(v, &endptr, 10);
    if (endptr == v || parsed == 0ull)
        return default_mb * 1024ull * 1024ull;
    return static_cast<size_t>(parsed) * 1024ull * 1024ull;
}

static cnpy::NpyArray get_npz_array(const cnpy::npz_t& npz_file, const std::vector<std::string>& candidate_names)
{
    for (const auto& name : candidate_names)
    {
        auto it = npz_file.find(name);
        if (it != npz_file.end())
            return it->second;
    }

    std::ostringstream oss;
    oss << "Missing NPZ key. Tried: ";
    for (size_t i = 0; i < candidate_names.size(); ++i)
    {
        oss << candidate_names[i];
        if (i + 1 < candidate_names.size())
            oss << ", ";
    }
    throw std::out_of_range(oss.str());
}

////////////////////////////////////////////// Data Generation

void generate_float_tensor(ggml_tensor* float_tensor, uint32_t float_tensor_size, uint32_t num_threads)
{
    float* data_ptr = (float*)float_tensor->data;

    std::mt19937 gen(233);

    uniform_real_distribution<float> dis(-5.0f, 5.0f);
    #pragma omp parallel for num_threads(num_threads)
    for(int i=0; i<float_tensor_size; ++i)
        data_ptr[i] = dis(gen);
}

void generate_lut_tensor(ggml_tensor* lut_table, uint32_t lut_table_size, uint32_t num_threads)
{
    lut_data_type* data_ptr = (lut_data_type*)lut_table->data;

    std::mt19937 gen(233);

    lut_data_type lower_bound = numeric_limits<lut_data_type>::min();
    lut_data_type upper_bound = numeric_limits<lut_data_type>::max();

    uniform_int_distribution<lut_data_type> dis(lower_bound, upper_bound);
    #pragma omp parallel for num_threads(num_threads)
    for(int i=0; i<lut_table_size; ++i)
        data_ptr[i] = dis(gen);
}

void generate_transformer_layer_weight(TransformerLayer& transformer_layer, TransformerParams& transformer_params)
{
    // generate attention norm weight
    transformer_layer.attention_norm = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.attention_params.token_dim);
    generate_float_tensor(transformer_layer.attention_norm, transformer_params.attention_params.token_dim, transformer_params.num_threads);
    transformer_layer.attention_norm_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.attention_params.token_dim);
    generate_float_tensor(transformer_layer.attention_norm_bias, transformer_params.attention_params.token_dim, transformer_params.num_threads);

#ifndef SEPARATE
    // generate qkv lut weight
    transformer_layer.qkv_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                        transformer_params.amm_param_list[0].index_params.sub_vec_len,
                                                        transformer_params.amm_param_list[0].index_params.num_centroid,
                                                        transformer_params.amm_param_list[0].index_params.num_codebook);
    uint32_t qkv_centroid_element_num = transformer_params.amm_param_list[0].index_params.sub_vec_len
                                      * transformer_params.amm_param_list[0].index_params.num_centroid
                                      * transformer_params.amm_param_list[0].index_params.num_codebook;
    generate_float_tensor(transformer_layer.qkv_centroid, qkv_centroid_element_num, transformer_params.num_threads);
    transformer_layer.qkv_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                         transformer_params.amm_param_list[0].lut_params.output_feature_len,
                                                         transformer_params.amm_param_list[0].lut_params.num_centroid,
                                                         transformer_params.amm_param_list[0].lut_params.num_codebook);
    uint32_t qkv_lut_element_num = transformer_params.amm_param_list[0].lut_params.output_feature_len
                                 * transformer_params.amm_param_list[0].lut_params.num_centroid
                                 * transformer_params.amm_param_list[0].lut_params.num_codebook;
    generate_lut_tensor(transformer_layer.qkv_lut_table, qkv_lut_element_num, transformer_params.num_threads);
    transformer_layer.qkv_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[0].lut_params.output_feature_len);
    uint32_t qkv_bias_element_num = transformer_params.amm_param_list[0].lut_params.output_feature_len;
    generate_float_tensor(transformer_layer.qkv_bias, qkv_bias_element_num, transformer_params.num_threads);
#else
    // generate q lut weight
    transformer_layer.q_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                      transformer_params.amm_param_list[4].index_params.sub_vec_len,
                                                      transformer_params.amm_param_list[4].index_params.num_centroid,
                                                      transformer_params.amm_param_list[4].index_params.num_codebook);
    uint32_t q_centroid_element_num = transformer_params.amm_param_list[4].index_params.sub_vec_len
                                    * transformer_params.amm_param_list[4].index_params.num_centroid
                                    * transformer_params.amm_param_list[4].index_params.num_codebook;
    generate_float_tensor(transformer_layer.q_centroid, q_centroid_element_num, transformer_params.num_threads);
    transformer_layer.q_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                       transformer_params.amm_param_list[4].lut_params.output_feature_len,
                                                       transformer_params.amm_param_list[4].lut_params.num_centroid,
                                                       transformer_params.amm_param_list[4].lut_params.num_codebook);
    uint32_t q_lut_element_num = transformer_params.amm_param_list[4].lut_params.output_feature_len
                               * transformer_params.amm_param_list[4].lut_params.num_centroid
                               * transformer_params.amm_param_list[4].lut_params.num_codebook;
    generate_lut_tensor(transformer_layer.q_lut_table, q_lut_element_num, transformer_params.num_threads);
    transformer_layer.q_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[4].lut_params.output_feature_len);
    uint32_t q_bias_element_num = transformer_params.amm_param_list[4].lut_params.output_feature_len;
    generate_float_tensor(transformer_layer.q_bias, q_bias_element_num, transformer_params.num_threads);

    // generate k lut weight
    transformer_layer.k_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                      transformer_params.amm_param_list[5].index_params.sub_vec_len,
                                                      transformer_params.amm_param_list[5].index_params.num_centroid,
                                                      transformer_params.amm_param_list[5].index_params.num_codebook);
    uint32_t k_centroid_element_num = transformer_params.amm_param_list[5].index_params.sub_vec_len
                                    * transformer_params.amm_param_list[5].index_params.num_centroid
                                    * transformer_params.amm_param_list[5].index_params.num_codebook;
    generate_float_tensor(transformer_layer.k_centroid, k_centroid_element_num, transformer_params.num_threads);
    transformer_layer.k_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                       transformer_params.amm_param_list[5].lut_params.output_feature_len,
                                                       transformer_params.amm_param_list[5].lut_params.num_centroid,
                                                       transformer_params.amm_param_list[5].lut_params.num_codebook);
    uint32_t k_lut_element_num = transformer_params.amm_param_list[5].lut_params.output_feature_len
                               * transformer_params.amm_param_list[5].lut_params.num_centroid
                               * transformer_params.amm_param_list[5].lut_params.num_codebook;
    generate_lut_tensor(transformer_layer.k_lut_table, k_lut_element_num, transformer_params.num_threads);
    transformer_layer.k_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[5].lut_params.output_feature_len);
    uint32_t k_bias_element_num = transformer_params.amm_param_list[5].lut_params.output_feature_len;
    generate_float_tensor(transformer_layer.k_bias, k_bias_element_num, transformer_params.num_threads);

    // generate v lut weight
    transformer_layer.v_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                      transformer_params.amm_param_list[6].index_params.sub_vec_len,
                                                      transformer_params.amm_param_list[6].index_params.num_centroid,
                                                      transformer_params.amm_param_list[6].index_params.num_codebook);
    uint32_t v_centroid_element_num = transformer_params.amm_param_list[6].index_params.sub_vec_len
                                    * transformer_params.amm_param_list[6].index_params.num_centroid
                                    * transformer_params.amm_param_list[6].index_params.num_codebook;
    generate_float_tensor(transformer_layer.v_centroid, v_centroid_element_num, transformer_params.num_threads);
    transformer_layer.v_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                       transformer_params.amm_param_list[6].lut_params.output_feature_len,
                                                       transformer_params.amm_param_list[6].lut_params.num_centroid,
                                                       transformer_params.amm_param_list[6].lut_params.num_codebook);
    uint32_t v_lut_element_num = transformer_params.amm_param_list[6].lut_params.output_feature_len
                               * transformer_params.amm_param_list[6].lut_params.num_centroid
                               * transformer_params.amm_param_list[6].lut_params.num_codebook;
    generate_lut_tensor(transformer_layer.v_lut_table, v_lut_element_num, transformer_params.num_threads);
    transformer_layer.v_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[6].lut_params.output_feature_len);
    uint32_t v_bias_element_num = transformer_params.amm_param_list[6].lut_params.output_feature_len;
    generate_float_tensor(transformer_layer.v_bias, v_bias_element_num, transformer_params.num_threads);
#endif

    // generate o lut weight
    transformer_layer.o_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                      transformer_params.amm_param_list[1].index_params.sub_vec_len,
                                                      transformer_params.amm_param_list[1].index_params.num_centroid,
                                                      transformer_params.amm_param_list[1].index_params.num_codebook);
    uint32_t o_centroid_element_num = transformer_params.amm_param_list[1].index_params.sub_vec_len
                                    * transformer_params.amm_param_list[1].index_params.num_centroid
                                    * transformer_params.amm_param_list[1].index_params.num_codebook;
    generate_float_tensor(transformer_layer.o_centroid, o_centroid_element_num, transformer_params.num_threads);
    transformer_layer.o_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                       transformer_params.amm_param_list[1].lut_params.output_feature_len,
                                                       transformer_params.amm_param_list[1].lut_params.num_centroid,
                                                       transformer_params.amm_param_list[1].lut_params.num_codebook);
    uint32_t o_lut_element_num = transformer_params.amm_param_list[1].lut_params.output_feature_len
                               * transformer_params.amm_param_list[1].lut_params.num_centroid
                               * transformer_params.amm_param_list[1].lut_params.num_codebook;
    generate_lut_tensor(transformer_layer.o_lut_table, o_lut_element_num, transformer_params.num_threads);
    transformer_layer.o_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[1].lut_params.output_feature_len);
    uint32_t o_bias_element_num = transformer_params.amm_param_list[1].lut_params.output_feature_len;
    generate_float_tensor(transformer_layer.o_bias, o_bias_element_num, transformer_params.num_threads);

    // generate ffn norm weight
    transformer_layer.ffn_norm = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.attention_params.token_dim);
    generate_float_tensor(transformer_layer.ffn_norm, transformer_params.attention_params.token_dim, transformer_params.num_threads);
    transformer_layer.ffn_norm_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.attention_params.token_dim);
    generate_float_tensor(transformer_layer.ffn_norm_bias, transformer_params.attention_params.token_dim, transformer_params.num_threads);


    // generate ffn1 lut weight
    transformer_layer.ffn1_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                         transformer_params.amm_param_list[2].index_params.sub_vec_len,
                                                         transformer_params.amm_param_list[2].index_params.num_centroid,
                                                         transformer_params.amm_param_list[2].index_params.num_codebook);
    uint32_t ffn1_centroid_element_num = transformer_params.amm_param_list[2].index_params.sub_vec_len
                                       * transformer_params.amm_param_list[2].index_params.num_centroid
                                       * transformer_params.amm_param_list[2].index_params.num_codebook;
    generate_float_tensor(transformer_layer.ffn1_centroid, ffn1_centroid_element_num, transformer_params.num_threads);
    transformer_layer.ffn1_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                          transformer_params.amm_param_list[2].lut_params.output_feature_len,
                                                          transformer_params.amm_param_list[2].lut_params.num_centroid,
                                                          transformer_params.amm_param_list[2].lut_params.num_codebook);
    uint32_t ffn1_lut_element_num = transformer_params.amm_param_list[2].lut_params.output_feature_len
                                  * transformer_params.amm_param_list[2].lut_params.num_centroid
                                  * transformer_params.amm_param_list[2].lut_params.num_codebook;
    generate_lut_tensor(transformer_layer.ffn1_lut_table, ffn1_lut_element_num, transformer_params.num_threads);
    transformer_layer.ffn1_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[2].lut_params.output_feature_len);
    uint32_t ffn1_bias_element_num = transformer_params.amm_param_list[2].lut_params.output_feature_len;
    generate_float_tensor(transformer_layer.ffn1_bias, ffn1_bias_element_num, transformer_params.num_threads);

    // generate ffn2 lut weight
    transformer_layer.ffn2_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                         transformer_params.amm_param_list[3].index_params.sub_vec_len,
                                                         transformer_params.amm_param_list[3].index_params.num_centroid,
                                                         transformer_params.amm_param_list[3].index_params.num_codebook);
    uint32_t ffn2_centroid_element_num = transformer_params.amm_param_list[3].index_params.sub_vec_len
                                       * transformer_params.amm_param_list[3].index_params.num_centroid
                                       * transformer_params.amm_param_list[3].index_params.num_codebook;
    generate_float_tensor(transformer_layer.ffn2_centroid, ffn2_centroid_element_num, transformer_params.num_threads);
    transformer_layer.ffn2_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                          transformer_params.amm_param_list[3].lut_params.output_feature_len,
                                                          transformer_params.amm_param_list[3].lut_params.num_centroid,
                                                          transformer_params.amm_param_list[3].lut_params.num_codebook);
    uint32_t ffn2_lut_element_num = transformer_params.amm_param_list[3].lut_params.output_feature_len
                                  * transformer_params.amm_param_list[3].lut_params.num_centroid
                                  * transformer_params.amm_param_list[3].lut_params.num_codebook;
    generate_lut_tensor(transformer_layer.ffn2_lut_table, ffn2_lut_element_num, transformer_params.num_threads);
    transformer_layer.ffn2_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[3].lut_params.output_feature_len);
    uint32_t ffn2_bias_element_num = transformer_params.amm_param_list[3].lut_params.output_feature_len;
    generate_float_tensor(transformer_layer.ffn2_bias, ffn2_bias_element_num, transformer_params.num_threads);
}

void load_float_tensor(ggml_tensor* float_tensor, uint32_t float_tensor_size, const cnpy::npz_t& npz_file, string float_tensor_name)
{
    cnpy::NpyArray npy_tensor = get_npz_array(npz_file, {float_tensor_name});
    float* npy_tensor_data = npy_tensor.data<float>();
    uint32_t npy_tensor_size = 1;
    for(int i=0; i<npy_tensor.shape.size(); ++i)
        npy_tensor_size *= npy_tensor.shape[i];
    assert(float_tensor_size==npy_tensor_size);
    memcpy(((float*)float_tensor->data), npy_tensor_data, sizeof(float)*float_tensor_size);
}

void load_lut_tensor(ggml_tensor* lut_table, uint32_t lut_table_size, const cnpy::npz_t& npz_file, string lut_table_name)
{
    cnpy::NpyArray npy_tensor = get_npz_array(npz_file, {lut_table_name});
    lut_data_type* npy_tensor_data = npy_tensor.data<lut_data_type>();
    uint32_t npy_tensor_size = 1;
    for(int i=0; i<npy_tensor.shape.size(); ++i)
        npy_tensor_size *= npy_tensor.shape[i];
    assert(lut_table_size==npy_tensor_size);
    memcpy(((lut_data_type*)lut_table->data), npy_tensor_data, sizeof(lut_data_type)*lut_table_size);
}

void load_layer_weight_from_file(uint32_t layer_id, TransformerLayer& transformer_layer, TransformerParams& transformer_params, const cnpy::npz_t& weight_npz)
{
    // generate attention norm weight
    transformer_layer.attention_norm = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.attention_params.token_dim);
    string attention_norm_name = to_string(layer_id) + ".attention.output.LayerNorm.weight";
    string attention_norm_name_alt = to_string(layer_id) + ".attention.output.LayerNorm.gamma";
    cnpy::NpyArray attention_norm_arr = get_npz_array(weight_npz, {attention_norm_name, attention_norm_name_alt});
    memcpy(((float*)transformer_layer.attention_norm->data), attention_norm_arr.data<float>(), sizeof(float)*transformer_params.attention_params.token_dim);
    transformer_layer.attention_norm_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.attention_params.token_dim);
    string attention_bias_name = to_string(layer_id) + ".attention.output.LayerNorm.bias";
    string attention_bias_name_alt = to_string(layer_id) + ".attention.output.LayerNorm.beta";
    cnpy::NpyArray attention_bias_arr = get_npz_array(weight_npz, {attention_bias_name, attention_bias_name_alt});
    memcpy(((float*)transformer_layer.attention_norm_bias->data), attention_bias_arr.data<float>(), sizeof(float)*transformer_params.attention_params.token_dim);

#ifndef SEPARATE
    // generate qkv lut weight
    transformer_layer.qkv_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                        transformer_params.amm_param_list[0].index_params.sub_vec_len,
                                                        transformer_params.amm_param_list[0].index_params.num_centroid,
                                                        transformer_params.amm_param_list[0].index_params.num_codebook);
    uint32_t qkv_centroid_element_num = transformer_params.amm_param_list[0].index_params.sub_vec_len
                                      * transformer_params.amm_param_list[0].index_params.num_centroid
                                      * transformer_params.amm_param_list[0].index_params.num_codebook;
    string qkv_centroid_name = to_string(layer_id) + ".attention.self.qkv.centroids.weight";
    load_float_tensor(transformer_layer.qkv_centroid, qkv_centroid_element_num, weight_npz, qkv_centroid_name);
    transformer_layer.qkv_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                         transformer_params.amm_param_list[0].lut_params.output_feature_len,
                                                         transformer_params.amm_param_list[0].lut_params.num_centroid,
                                                         transformer_params.amm_param_list[0].lut_params.num_codebook);
    uint32_t qkv_lut_element_num = transformer_params.amm_param_list[0].lut_params.output_feature_len
                                 * transformer_params.amm_param_list[0].lut_params.num_centroid
                                 * transformer_params.amm_param_list[0].lut_params.num_codebook;
    string qkv_lut_name = to_string(layer_id) + ".attention.self.qkv.lut.tensor";
    load_lut_tensor(transformer_layer.qkv_lut_table, qkv_lut_element_num, weight_npz, qkv_lut_name);
    transformer_layer.qkv_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[0].lut_params.output_feature_len);
    uint32_t qkv_bias_element_num = transformer_params.amm_param_list[0].lut_params.output_feature_len;
    string qkv_bias_bame = to_string(layer_id) + ".attention.self.qkv.bias";
    load_float_tensor(transformer_layer.qkv_bias, qkv_bias_element_num, weight_npz, qkv_bias_bame);
    string qkv_lut_scale_name = to_string(layer_id) + ".attention.self.qkv.lut.scale";
    transformer_params.amm_param_list[0].lut_params.scale = weight_npz.at(qkv_lut_scale_name).data<float>()[0];
    string qkv_lut_bias_name = to_string(layer_id) + ".attention.self.qkv.lut.bias";
    transformer_params.amm_param_list[0].lut_params.bias = weight_npz.at(qkv_lut_bias_name).data<float>()[0];
#else
    // generate q lut weight
    transformer_layer.q_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                      transformer_params.amm_param_list[4].index_params.sub_vec_len,
                                                      transformer_params.amm_param_list[4].index_params.num_centroid,
                                                      transformer_params.amm_param_list[4].index_params.num_codebook);
    uint32_t q_centroid_element_num = transformer_params.amm_param_list[4].index_params.sub_vec_len
                                    * transformer_params.amm_param_list[4].index_params.num_centroid
                                    * transformer_params.amm_param_list[4].index_params.num_codebook;
    string q_centroid_name = to_string(layer_id) + ".attention.self.query.centroids.weight";
    load_float_tensor(transformer_layer.q_centroid, q_centroid_element_num, weight_npz, q_centroid_name);
    transformer_layer.q_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                       transformer_params.amm_param_list[4].lut_params.output_feature_len,
                                                       transformer_params.amm_param_list[4].lut_params.num_centroid,
                                                       transformer_params.amm_param_list[4].lut_params.num_codebook);
    uint32_t q_lut_element_num = transformer_params.amm_param_list[4].lut_params.output_feature_len
                               * transformer_params.amm_param_list[4].lut_params.num_centroid
                               * transformer_params.amm_param_list[4].lut_params.num_codebook;
    string q_lut_name = to_string(layer_id) + ".attention.self.query.lut.tensor";
    load_lut_tensor(transformer_layer.q_lut_table, q_lut_element_num, weight_npz, q_lut_name);
    transformer_layer.q_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[4].lut_params.output_feature_len);
    uint32_t q_bias_element_num = transformer_params.amm_param_list[4].lut_params.output_feature_len;
    string q_bias_bame = to_string(layer_id) + ".attention.self.query.bias";
    load_float_tensor(transformer_layer.q_bias, q_bias_element_num, weight_npz, q_bias_bame);
    string q_lut_scale_name = to_string(layer_id) + ".attention.self.query.lut.scale";
    transformer_params.amm_param_list[4].lut_params.scale = weight_npz.at(q_lut_scale_name).data<float>()[0];
    string q_lut_bias_name = to_string(layer_id) + ".attention.self.query.lut.bias";
    transformer_params.amm_param_list[4].lut_params.bias = weight_npz.at(q_lut_bias_name).data<float>()[0];

    // generate k lut weight
    transformer_layer.k_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                      transformer_params.amm_param_list[5].index_params.sub_vec_len,
                                                      transformer_params.amm_param_list[5].index_params.num_centroid,
                                                      transformer_params.amm_param_list[5].index_params.num_codebook);
    uint32_t k_centroid_element_num = transformer_params.amm_param_list[5].index_params.sub_vec_len
                                    * transformer_params.amm_param_list[5].index_params.num_centroid
                                    * transformer_params.amm_param_list[5].index_params.num_codebook;
    string k_centroid_name = to_string(layer_id) + ".attention.self.key.centroids.weight";
    load_float_tensor(transformer_layer.k_centroid, k_centroid_element_num, weight_npz, k_centroid_name);
    transformer_layer.k_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                       transformer_params.amm_param_list[5].lut_params.output_feature_len,
                                                       transformer_params.amm_param_list[5].lut_params.num_centroid,
                                                       transformer_params.amm_param_list[5].lut_params.num_codebook);
    uint32_t k_lut_element_num = transformer_params.amm_param_list[5].lut_params.output_feature_len
                               * transformer_params.amm_param_list[5].lut_params.num_centroid
                               * transformer_params.amm_param_list[5].lut_params.num_codebook;
    string k_lut_name = to_string(layer_id) + ".attention.self.key.lut.tensor";
    load_lut_tensor(transformer_layer.k_lut_table, k_lut_element_num, weight_npz, k_lut_name);
    transformer_layer.k_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[5].lut_params.output_feature_len);
    uint32_t k_bias_element_num = transformer_params.amm_param_list[5].lut_params.output_feature_len;
    string k_bias_bame = to_string(layer_id) + ".attention.self.key.bias";
    load_float_tensor(transformer_layer.k_bias, k_bias_element_num, weight_npz, k_bias_bame);
    string k_lut_scale_name = to_string(layer_id) + ".attention.self.key.lut.scale";
    transformer_params.amm_param_list[5].lut_params.scale = weight_npz.at(k_lut_scale_name).data<float>()[0];
    string k_lut_bias_name = to_string(layer_id) + ".attention.self.key.lut.bias";
    transformer_params.amm_param_list[5].lut_params.bias = weight_npz.at(k_lut_bias_name).data<float>()[0];

    // generate v lut weight
    transformer_layer.v_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                      transformer_params.amm_param_list[6].index_params.sub_vec_len,
                                                      transformer_params.amm_param_list[6].index_params.num_centroid,
                                                      transformer_params.amm_param_list[6].index_params.num_codebook);
    uint32_t v_centroid_element_num = transformer_params.amm_param_list[6].index_params.sub_vec_len
                                    * transformer_params.amm_param_list[6].index_params.num_centroid
                                    * transformer_params.amm_param_list[6].index_params.num_codebook;
    string v_centroid_name = to_string(layer_id) + ".attention.self.value.centroids.weight";
    load_float_tensor(transformer_layer.v_centroid, v_centroid_element_num, weight_npz, v_centroid_name);
    transformer_layer.v_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                       transformer_params.amm_param_list[6].lut_params.output_feature_len,
                                                       transformer_params.amm_param_list[6].lut_params.num_centroid,
                                                       transformer_params.amm_param_list[6].lut_params.num_codebook);
    uint32_t v_lut_element_num = transformer_params.amm_param_list[6].lut_params.output_feature_len
                               * transformer_params.amm_param_list[6].lut_params.num_centroid
                               * transformer_params.amm_param_list[6].lut_params.num_codebook;
    string v_lut_name = to_string(layer_id) + ".attention.self.value.lut.tensor";
    load_lut_tensor(transformer_layer.v_lut_table, v_lut_element_num, weight_npz, v_lut_name);
    transformer_layer.v_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[6].lut_params.output_feature_len);
    uint32_t v_bias_element_num = transformer_params.amm_param_list[6].lut_params.output_feature_len;
    string v_bias_bame = to_string(layer_id) + ".attention.self.value.bias";
    load_float_tensor(transformer_layer.v_bias, v_bias_element_num, weight_npz, v_bias_bame);
    string v_lut_scale_name = to_string(layer_id) + ".attention.self.value.lut.scale";
    transformer_params.amm_param_list[6].lut_params.scale = weight_npz.at(v_lut_scale_name).data<float>()[0];
    string v_lut_bias_name = to_string(layer_id) + ".attention.self.value.lut.bias";
    transformer_params.amm_param_list[6].lut_params.bias = weight_npz.at(v_lut_bias_name).data<float>()[0];
#endif

    // generate o lut weight
    transformer_layer.o_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                      transformer_params.amm_param_list[1].index_params.sub_vec_len,
                                                      transformer_params.amm_param_list[1].index_params.num_centroid,
                                                      transformer_params.amm_param_list[1].index_params.num_codebook);
    uint32_t o_centroid_element_num = transformer_params.amm_param_list[1].index_params.sub_vec_len
                                    * transformer_params.amm_param_list[1].index_params.num_centroid
                                    * transformer_params.amm_param_list[1].index_params.num_codebook;
    string o_centroid_name = to_string(layer_id) + ".attention.output.dense.centroids.weight";
    load_float_tensor(transformer_layer.o_centroid, o_centroid_element_num, weight_npz, o_centroid_name);
    transformer_layer.o_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                       transformer_params.amm_param_list[1].lut_params.output_feature_len,
                                                       transformer_params.amm_param_list[1].lut_params.num_centroid,
                                                       transformer_params.amm_param_list[1].lut_params.num_codebook);
    uint32_t o_lut_element_num = transformer_params.amm_param_list[1].lut_params.output_feature_len
                               * transformer_params.amm_param_list[1].lut_params.num_centroid
                               * transformer_params.amm_param_list[1].lut_params.num_codebook;
    string o_lut_name = to_string(layer_id) + ".attention.output.dense.lut.tensor";
    load_lut_tensor(transformer_layer.o_lut_table, o_lut_element_num, weight_npz, o_lut_name);
    transformer_layer.o_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[1].lut_params.output_feature_len);
    uint32_t o_bias_element_num = transformer_params.amm_param_list[1].lut_params.output_feature_len;
    string o_bias_bame = to_string(layer_id) + ".attention.output.dense.bias";
    load_float_tensor(transformer_layer.o_bias, o_bias_element_num, weight_npz, o_bias_bame);
    string o_lut_scale_name = to_string(layer_id) + ".attention.output.dense.lut.scale";
    transformer_params.amm_param_list[1].lut_params.scale = weight_npz.at(o_lut_scale_name).data<float>()[0];
    string o_lut_bias_name = to_string(layer_id) + ".attention.output.dense.lut.bias";
    transformer_params.amm_param_list[1].lut_params.bias = weight_npz.at(o_lut_bias_name).data<float>()[0];

    // generate ffn norm weight
    transformer_layer.ffn_norm = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.attention_params.token_dim);
    string ffn_norm_name = to_string(layer_id) + ".output.LayerNorm.weight";
    string ffn_norm_name_alt = to_string(layer_id) + ".output.LayerNorm.gamma";
    cnpy::NpyArray ffn_norm_arr = get_npz_array(weight_npz, {ffn_norm_name, ffn_norm_name_alt});
    memcpy(((float*)transformer_layer.ffn_norm->data), ffn_norm_arr.data<float>(), sizeof(float)*transformer_params.attention_params.token_dim);
    transformer_layer.ffn_norm_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.attention_params.token_dim);
    string ffn_bias_name = to_string(layer_id) + ".output.LayerNorm.bias";
    string ffn_bias_name_alt = to_string(layer_id) + ".output.LayerNorm.beta";
    cnpy::NpyArray ffn_bias_arr = get_npz_array(weight_npz, {ffn_bias_name, ffn_bias_name_alt});
    memcpy(((float*)transformer_layer.ffn_norm_bias->data), ffn_bias_arr.data<float>(), sizeof(float)*transformer_params.attention_params.token_dim);
    
    // generate ffn1 lut weight
    transformer_layer.ffn1_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                         transformer_params.amm_param_list[2].index_params.sub_vec_len,
                                                         transformer_params.amm_param_list[2].index_params.num_centroid,
                                                         transformer_params.amm_param_list[2].index_params.num_codebook);
    uint32_t ffn1_centroid_element_num = transformer_params.amm_param_list[2].index_params.sub_vec_len
                                       * transformer_params.amm_param_list[2].index_params.num_centroid
                                       * transformer_params.amm_param_list[2].index_params.num_codebook;
    string ffn1_centroid_name = to_string(layer_id) + ".intermediate.dense.centroids.weight";
    load_float_tensor(transformer_layer.ffn1_centroid, ffn1_centroid_element_num, weight_npz, ffn1_centroid_name);
    transformer_layer.ffn1_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                          transformer_params.amm_param_list[2].lut_params.output_feature_len,
                                                          transformer_params.amm_param_list[2].lut_params.num_centroid,
                                                          transformer_params.amm_param_list[2].lut_params.num_codebook);
    uint32_t ffn1_lut_element_num = transformer_params.amm_param_list[2].lut_params.output_feature_len
                                  * transformer_params.amm_param_list[2].lut_params.num_centroid
                                  * transformer_params.amm_param_list[2].lut_params.num_codebook;
    string ffn1_lut_name = to_string(layer_id) + ".intermediate.dense.lut.tensor";
    load_lut_tensor(transformer_layer.ffn1_lut_table, ffn1_lut_element_num, weight_npz, ffn1_lut_name);
    transformer_layer.ffn1_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[2].lut_params.output_feature_len);
    uint32_t ffn1_bias_element_num = transformer_params.amm_param_list[2].lut_params.output_feature_len;
    string ffn1_bias_bame = to_string(layer_id) + ".intermediate.dense.bias";
    load_float_tensor(transformer_layer.ffn1_bias, ffn1_bias_element_num, weight_npz, ffn1_bias_bame);
    string ffn1_lut_scale_name = to_string(layer_id) + ".intermediate.dense.lut.scale";
    transformer_params.amm_param_list[2].lut_params.scale = weight_npz.at(ffn1_lut_scale_name).data<float>()[0];
    string ffn1_lut_bias_name = to_string(layer_id) + ".intermediate.dense.lut.bias";
    transformer_params.amm_param_list[2].lut_params.bias = weight_npz.at(ffn1_lut_bias_name).data<float>()[0];

    // generate ffn2 lut weight
    transformer_layer.ffn2_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                         transformer_params.amm_param_list[3].index_params.sub_vec_len,
                                                         transformer_params.amm_param_list[3].index_params.num_centroid,
                                                         transformer_params.amm_param_list[3].index_params.num_codebook);
    uint32_t ffn2_centroid_element_num = transformer_params.amm_param_list[3].index_params.sub_vec_len
                                       * transformer_params.amm_param_list[3].index_params.num_centroid
                                       * transformer_params.amm_param_list[3].index_params.num_codebook;
    string ffn2_centroid_name = to_string(layer_id) + ".output.dense.centroids.weight";
    load_float_tensor(transformer_layer.ffn2_centroid, ffn2_centroid_element_num, weight_npz, ffn2_centroid_name);
    transformer_layer.ffn2_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                          transformer_params.amm_param_list[3].lut_params.output_feature_len,
                                                          transformer_params.amm_param_list[3].lut_params.num_centroid,
                                                          transformer_params.amm_param_list[3].lut_params.num_codebook);
    uint32_t ffn2_lut_element_num = transformer_params.amm_param_list[3].lut_params.output_feature_len
                                  * transformer_params.amm_param_list[3].lut_params.num_centroid
                                  * transformer_params.amm_param_list[3].lut_params.num_codebook;
    string ffn2_lut_name = to_string(layer_id) + ".output.dense.lut.tensor";
    load_lut_tensor(transformer_layer.ffn2_lut_table, ffn2_lut_element_num, weight_npz, ffn2_lut_name);
    transformer_layer.ffn2_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[3].lut_params.output_feature_len);
    uint32_t ffn2_bias_element_num = transformer_params.amm_param_list[3].lut_params.output_feature_len;
    string ffn2_bias_bame = to_string(layer_id) + ".output.dense.bias";
    load_float_tensor(transformer_layer.ffn2_bias, ffn2_bias_element_num, weight_npz, ffn2_bias_bame);
    string ffn2_lut_scale_name = to_string(layer_id) + ".output.dense.lut.scale";
    transformer_params.amm_param_list[3].lut_params.scale = weight_npz.at(ffn2_lut_scale_name).data<float>()[0];
    string ffn2_lut_bias_name = to_string(layer_id) + ".output.dense.lut.bias";
    transformer_params.amm_param_list[3].lut_params.bias = weight_npz.at(ffn2_lut_bias_name).data<float>()[0];  
}

void generate_transformer_model_weight(Transformer& transformer, TransformerParams& transformer_params, string model_weight_path)
{
    transformer.layer_num = transformer_params.layer_num;
    transformer.layer_params.resize(transformer.layer_num);

    ifstream weight_file(model_weight_path);
    if(weight_file.is_open())
    {
        printf("weight file is found, load weights from file.\n");
        cnpy::npz_t weight_npz = cnpy::npz_load(model_weight_path);
        for(int i=0; i<transformer.layer_num; ++i)
        {
            transformer.layer_params[i] = new TransformerLayer;
            transformer.layer_params[i]->ctx = transformer.ctx;
            load_layer_weight_from_file(i, *transformer.layer_params[i], transformer_params, weight_npz);
        }
    }
    else
    {
        printf("weight file is not found, generate weights randomly.\n");
        for(int i=0; i<transformer.layer_num; ++i)
        {
            transformer.layer_params[i] = new TransformerLayer;
            transformer.layer_params[i]->ctx = transformer.ctx;
            generate_transformer_layer_weight(*transformer.layer_params[i], transformer_params);
        }
    }
}

////////////////////////////////////////////// Main function entry

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        printf("usage: %s config_yaml_path weight_path\n", argv[0]);
        exit(-1);
    }
    string config_yaml_path(argv[1]);
    string model_weight_path(argv[2]);

    ////////////////////////////////////////////// transformer param parsing

    // dump transformer params
    TransformerParams transformer_params;
    parse_transformer_configs(transformer_params, config_yaml_path);
    printf("finish parameter parsing\n");

    ////////////////////////////////////////////// transformer weight init

    // init transformer weight
    Transformer transformer;
    size_t    model_size = get_env_size_mb_or_default("PIMDL_MODEL_CTX_MB", 2048);
    uint8_t*  model_addr = new uint8_t[model_size];
    struct ggml_init_params model_ctx_params = {
            /*.mem_size   =*/ model_size,
            /*.mem_buffer =*/ model_addr,
            /*.no_alloc   =*/ false,
    };
    transformer.ctx = ggml_init(model_ctx_params);
    generate_transformer_model_weight(transformer, transformer_params, model_weight_path);
    printf("finish weight initialization\n");

    ////////////////////////////////////////////// DPU allocate

    dpu_set_t dpu_set;
    uint32_t allocated_dpu_num;
    if(transformer_params.attention_params.dpu_num == NR_DPUS_PER_RANK)
        allocated_dpu_num = allocate_rank(&dpu_set, transformer_params.attention_params.dpu_num / NR_DPUS_PER_RANK);
    else
        allocated_dpu_num = allocate_dpu(&dpu_set, transformer_params.attention_params.dpu_num);
    printf("finish dpu allocation\n");

    ////////////////////////////////////////////// inference

    size_t    data_size = get_env_size_mb_or_default("PIMDL_DATA_CTX_MB", 2048);
    uint8_t*  data_addr = new uint8_t[data_size];
    struct ggml_init_params data_ctx_params = {
            /*.mem_size   =*/ data_size,
            /*.mem_buffer =*/ data_addr,
            /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx0 = ggml_init(data_ctx_params);
    struct ggml_tensor* input_tensor = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, transformer_params.attention_params.token_dim, transformer_params.attention_params.n);
    generate_float_tensor(input_tensor, transformer_params.attention_params.n * transformer_params.attention_params.token_dim, transformer_params.num_threads);

    printf("inference start\n");
    double time1 = W_time();
    struct ggml_tensor* model_output = transformer_inference(&dpu_set, ctx0, &transformer, input_tensor, transformer_params);
    double time2 = W_time();
    printf("inference end\n");

    double elapsed_s = time2 - time1;
    double elapsed_ms = elapsed_s * 1000.0;
    double tokens_per_s = transformer_params.attention_params.n / elapsed_s;
    double per_layer_ms = elapsed_ms / transformer_params.layer_num;

    printf("e2e_inference_time_s %.6f\n", elapsed_s);
    printf("e2e_inference_time_ms %.6f\n", elapsed_ms);
    printf("e2e_tokens_per_s %.6f\n", tokens_per_s);
    printf("avg_layer_time_ms %.6f\n", per_layer_ms);

    // Keep output tensor alive and touch one element to avoid accidental optimization.
    printf("output_sample %.6f\n", ((float*)model_output->data)[0]);
}
