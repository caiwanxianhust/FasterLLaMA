set (FASTER_LLAMA_HEADER_DIR
    ${PROJECT_SOURCE_DIR}/fasterLlama
    ${PROJECT_SOURCE_DIR}/fasterLlama/cuda
)

include_directories(
    ${FASTER_LLAMA_HEADER_DIR}
)

add_subdirectory(cuda)

