set (sample_fasterllama_fp32_files
    llama_fp32.cu
)

set (sample_fasterllama_fp16_files
    llama_fp16.cu
)

set (SAMPLES_HEADER_DIR
    ${PROJECT_SOURCE_DIR}/fasterLlama
    ${PROJECT_SOURCE_DIR}/fasterLlama/cuda
)

include_directories(
    ${SAMPLES_HEADER_DIR}
)

link_directories(
    ${PROJECT_SOURCE_DIR}/fasterLlama/lib
)

message("-- Assign include directories (include_directories=${SAMPLES_HEADER_DIR})")

add_executable(fasterllama_fp32 ${sample_fasterllama_fp32_files})
target_compile_features(fasterllama_fp32 PUBLIC cxx_std_14)
target_link_libraries(fasterllama_fp32 PUBLIC -lcublas -lcudart fldecodersampling)

add_executable(fasterllama_fp16 ${sample_fasterllama_fp16_files})
target_compile_features(fasterllama_fp16 PUBLIC cxx_std_14)
target_link_libraries(fasterllama_fp16 PUBLIC -lcublas -lcudart fldecodersampling)
