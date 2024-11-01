set(decoder_kernel_files
    decoder_kernels.cu
)

set(decoding_kernel_files
    decoding_kernels.cu
)

set(open_decoder_files
    open_decoder.cu
)

set(decoding_sampling_files
    decoding_sampling.cu
)

set(FASTER_LLAMA_CUDA_HEADER_DIR
    ${PROJECT_SOURCE_DIR}/fasterLlama
    ${PROJECT_SOURCE_DIR}/fasterLlama/cuda
)

include_directories(
    ${FASTER_LLAMA_CUDA_HEADER_DIR}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/fasterLlama/lib)

message("-- Assign fasterLlama include directories (include_directories=${FASTER_LLAMA_CUDA_HEADER_DIR})")
message("-- in fasterLlama cuda Assign arch (arch=${CMAKE_CUDA_ARCHITECTURES})")

add_library(fldecoderkernel SHARED ${decoder_kernel_files})
# Request that particles be built with --std=c++14
# As this is a public compile feature anything that links to particles
# will also build with -std=c++14
target_compile_features(fldecoderkernel PUBLIC cxx_std_14)
target_compile_options(fldecoderkernel PUBLIC "-gencode=arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}")
set_target_properties(fldecoderkernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(fldecoderkernel PUBLIC -lcublas -lcudart -lcurand)

add_library(fldecodingkernel SHARED ${decoding_kernel_files})
# Request that particles be built with --std=c++14
# As this is a public compile feature anything that links to particles
# will also build with -std=c++14
target_compile_features(fldecodingkernel PUBLIC cxx_std_14)
target_compile_options(fldecodingkernel PUBLIC "-gencode=arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}")
set_target_properties(fldecodingkernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(fldecodingkernel PUBLIC -lcublas -lcudart -lcurand)

add_library(flopendecoder SHARED ${open_decoder_files})
# Request that particles be built with --std=c++14
# As this is a public compile feature anything that links to particles
# will also build with -std=c++14
target_compile_features(flopendecoder PUBLIC cxx_std_14)
target_compile_options(flopendecoder PUBLIC "-gencode=arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}")
set_target_properties(flopendecoder PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(flopendecoder PUBLIC -lcublas -lcudart -lcurand fldecoderkernel)

add_library(fldecodersampling SHARED ${decoding_sampling_files})
# Request that particles be built with --std=c++14
# As this is a public compile feature anything that links to particles
# will also build with -std=c++14
target_compile_features(fldecodersampling PUBLIC cxx_std_14)
target_compile_options(fldecodersampling PUBLIC "-gencode=arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}")
set_target_properties(fldecodersampling PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(fldecodersampling PUBLIC -lcublas -lcudart -lcurand flopendecoder fldecodingkernel fldecoderkernel)