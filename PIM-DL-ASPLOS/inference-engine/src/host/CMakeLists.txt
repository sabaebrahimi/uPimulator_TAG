set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g -O3 ")

message("CMAKE CXX FLAGS: " ${CMAKE_CXX_FLAGS})

add_library(host SHARED   
    amm_host.cpp
    index_calc_host.cpp
    parser.cpp
    pim_lut_host.cpp
    transformer_layer.cpp
    transformer_model.cpp
)

target_compile_features(host PUBLIC cxx_std_11)
target_link_libraries(host ${DPU_LIBRARIES} ggml yaml-cpp)
if(DEBUG_MODE)
    target_compile_definitions(host PUBLIC DEBUG)
endif()
if(AMM_BREAKDOWN)
    target_compile_definitions(host PUBLIC AMM_BREAKDOWN)
endif()
if(TRANSFORMER_BREAKDOWN)
    target_compile_definitions(host PUBLIC TRANSFORMER_BREAKDOWN)
endif()
if(TRANSFORMER_BREAKDOWN_AMM)
    target_compile_definitions(host PUBLIC TRANSFORMER_BREAKDOWN_AMM)
endif()
if(CNN_BREAKDOWN_AMM)
    target_compile_definitions(host PUBLIC CNN_BREAKDOWN_AMM)
endif()
# uPIMulator co-simulation emitter: dump per-DPU LUT/index/output snapshots at
# each projection boundary. Enable with -DPIMDL_COSIM_DUMP=ON. See COSIM_README.md.
if(PIMDL_COSIM_DUMP)
    target_compile_definitions(host PUBLIC PIMDL_COSIM_DUMP)
endif()
