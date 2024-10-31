#include "decoding_sampling.h"
#include <cstdio>


template <typename T>
void printVecInVec(const T *clusters, const int nrows, const int ncols, const int end_row, const int end_col, const char *str)
{
    printf("%s:\n[\n", str);
    for (int i = 0; i < end_row; ++i)
    {
        printf("[");
        for (int j = 0; j < end_col; ++j)
        {
            printf("%g  ", static_cast<float>(clusters[i * ncols + j]));
        }
        printf("]\n");
    }
    printf("]\n");
}

template <>
void printVecInVec(const half *clusters, const int nrows, const int ncols, const int end_row, const int end_col, const char *str)
{
    printf("%s:\n[\n", str);
    if (end_row >= nrows || end_col >= ncols)
        printf("invalid arguments!!!\nend_row >= nrows or end_col >= ncols\n");
    for (int i = 0; i < end_row; ++i)
    {
        printf("[");
        for (int j = 0; j < end_col; ++j)
        {
            printf("%g  ", __half2float(clusters[i * ncols + j]));
        }
        printf("]\n");
    }
    printf("]\n");
}

template <typename T>
void device_malloc(T **ptr, int size)
{
    CHECK_CUDA_ERROR(cudaMalloc((void **)ptr, sizeof(T) * size));
    T *tmp = new T[size];
    for (int i = 0; i < size; i++)
        tmp[i] = (T)((float)rand() / (RAND_MAX + 1.0) * 0.02);
    CHECK_CUDA_ERROR(cudaMemcpy(*ptr, tmp, sizeof(T) * size, cudaMemcpyHostToDevice));
    delete[] tmp;
}

template <typename T>
__global__ void initAttnMaskKernel(T *attn_mask, const int length)
{
    int row = blockIdx.x;
    for (int tid = threadIdx.x; tid < length; tid += blockDim.x)
    {
        attn_mask[row * length + tid] = (row > tid) ? 0.0f : (T)(-1 * FLT_MAX);
    }
}

__global__ void initIntVecKernel(int *mat, const int length, const int max_val, const int min_val)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length)
    {
        curandState_t local_state;
        curand_init(0, tid, 0, &local_state);
        int val = static_cast<int>(curand(&local_state) % (max_val - min_val));
        mat[tid] = val + min_val;
    }
}

template <typename T>
void decoding_sample(const int batch_size, const int candidate_num, const float probability_threshold, const int head_num,
                     const int size_per_head, const int vocab_size, const int max_prompt_len, const int max_gen_len,
                     const int decoder_layers, const int ffn_hidden_units)
{
    const int hidden_units = head_num * size_per_head;
    const int total_len = max_prompt_len + max_gen_len;
    const int end_id = 2;

    cublasHandle_t cublasHandle;
    CHECK_CUBLAS_STATUS(cublasCreate(&cublasHandle));

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUBLAS_STATUS(cublasSetStream(cublasHandle, stream));

    tinycudallama::Allocator<tinycudallama::AllocatorType::CUDA> allocator(0);
    tinycudallama::DecoderInitParam<T, int8_t> *param = new tinycudallama::DecoderInitParam<T, int8_t>[decoder_layers];

    for (int i = 0; i < decoder_layers; i++)
    {
        param[i].stream = stream;
        param[i].cublas_handle = cublasHandle;

        T *d_attn_resnorm_gamma;
        int8_t *d_self_Q_kernel, *d_self_K_kernel, *d_self_V_kernel, *d_self_output_kernel;
        float *d_self_Q_kernel_scale, *d_self_K_kernel_scale, *d_self_V_kernel_scale, *d_self_output_kernel_scale;
        float *d_attn_mask; // [total_len, total_len]
        T *d_ffn_resnorm_gamma;
        int8_t *d_ffn_kernel1, *d_ffn_kernel2, *d_ffn_kernel3;
        float *d_ffn_kernel1_scale, *d_ffn_kernel2_scale, *d_ffn_kernel3_scale;

        device_malloc(&d_attn_resnorm_gamma, hidden_units);

        device_malloc(&d_self_Q_kernel, hidden_units * hidden_units);
        device_malloc(&d_self_K_kernel, hidden_units * hidden_units);
        device_malloc(&d_self_V_kernel, hidden_units * hidden_units);
        device_malloc(&d_self_output_kernel, hidden_units * hidden_units);
        device_malloc(&d_self_Q_kernel_scale, hidden_units);
        device_malloc(&d_self_K_kernel_scale, hidden_units);
        device_malloc(&d_self_V_kernel_scale, hidden_units);
        device_malloc(&d_self_output_kernel_scale, hidden_units);

        // attn_mask 为下三角为 0，其他元素为 -lnf 的矩阵，各层复用
        if (i == 0)
        {
            device_malloc(&d_attn_mask, total_len * total_len);
            initAttnMaskKernel<<<total_len, 256, 0, stream>>>(d_attn_mask, total_len);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR(cudaGetLastError());
            param[i].attn_mask = d_attn_mask;
        }
        else
        {
            param[i].attn_mask = param[i - 1].attn_mask;
        }

        device_malloc(&d_ffn_resnorm_gamma, hidden_units);

        device_malloc(&d_ffn_kernel1, ffn_hidden_units * hidden_units);
        device_malloc(&d_ffn_kernel1_scale, ffn_hidden_units);
        device_malloc(&d_ffn_kernel2, ffn_hidden_units * hidden_units);
        device_malloc(&d_ffn_kernel2_scale, hidden_units);
        device_malloc(&d_ffn_kernel3, ffn_hidden_units * hidden_units);
        device_malloc(&d_ffn_kernel3_scale, ffn_hidden_units);

        param[i].attn_resnorm.gamma = d_attn_resnorm_gamma;
        param[i].attn_resnorm.eps = 1e-5f;
        param[i].attention.query_weight.kernel = d_self_Q_kernel;
        param[i].attention.query_weight.weight_scale = d_self_Q_kernel_scale;
        param[i].attention.key_weight.kernel = d_self_K_kernel;
        param[i].attention.key_weight.weight_scale = d_self_K_kernel_scale;
        param[i].attention.value_weight.kernel = d_self_V_kernel;
        param[i].attention.value_weight.weight_scale = d_self_Q_kernel_scale;
        param[i].attention.attention_output_weight.kernel = d_self_output_kernel;
        param[i].attention.attention_output_weight.weight_scale = d_self_output_kernel_scale;
        param[i].ffn_resnorm.gamma = d_ffn_resnorm_gamma;
        param[i].ffn_resnorm.eps = 1e-5f;
        param[i].ffn.w1_weight.kernel = d_ffn_kernel1;
        param[i].ffn.w1_weight.weight_scale = d_ffn_kernel1_scale;
        param[i].ffn.w2_weight.kernel = d_ffn_kernel2;
        param[i].ffn.w2_weight.weight_scale = d_ffn_kernel2_scale;
        param[i].ffn.w3_weight.kernel = d_ffn_kernel3;
        param[i].ffn.w3_weight.weight_scale = d_ffn_kernel3_scale;
    }

    tinycudallama::DecodingInitParam<T> decoding_params;

    float *d_embedding_table;
    float *d_freq_cis;
    int *d_prompt_sequence_length;
    int *d_prompt_tokens;
    bool *d_prompt_tokens_mask;
    T *d_decoding_resnorm_gamma;
    T *d_output_weight_kernel;
    int *d_output_ids;
    int *d_sequence_lengths;

    device_malloc(&d_embedding_table, hidden_units * vocab_size);
    device_malloc(&d_freq_cis, total_len * size_per_head);
    device_malloc(&d_prompt_sequence_length, batch_size);
    device_malloc(&d_prompt_tokens, max_prompt_len * batch_size);
    device_malloc(&d_prompt_tokens_mask, max_prompt_len * batch_size);
    device_malloc(&d_decoding_resnorm_gamma, hidden_units);
    device_malloc(&d_output_weight_kernel, hidden_units * vocab_size);
    device_malloc(&d_output_ids, batch_size * total_len);
    device_malloc(&d_sequence_lengths, batch_size);

    int *h_prompt_sequence_length = new int[batch_size];
    bool *h_prompt_tokens_mask = new bool[max_prompt_len * batch_size];
    int min_prompt_seq_len = INT_MAX;
    int max_prompt_seq_len = -1;
    for (int i = 0; i < batch_size; i++)
    {
        h_prompt_sequence_length[i] = max_prompt_len - batch_size + (rand() % batch_size) + 1;
        min_prompt_seq_len = min(min_prompt_seq_len, h_prompt_sequence_length[i]);
        max_prompt_seq_len = max(max_prompt_seq_len, h_prompt_sequence_length[i]);
        for (int j = 0; j < max_prompt_len; ++j)
        {
            h_prompt_tokens_mask[i * max_prompt_len + j] = (j < h_prompt_sequence_length[i]);
        }
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_prompt_sequence_length, h_prompt_sequence_length, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_prompt_tokens_mask, h_prompt_tokens_mask, sizeof(bool) * max_prompt_len * batch_size, cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = (max_prompt_len * batch_size + block_size - 1) / block_size;
    initIntVecKernel<<<grid_size, block_size>>>(d_prompt_tokens, max_prompt_len * batch_size, vocab_size, 3);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError());

    int *h_prompt_tokens = new int[max_prompt_len * batch_size];
    CHECK_CUDA_ERROR(cudaMemcpy(h_prompt_tokens, d_prompt_tokens, sizeof(int) * max_prompt_len * batch_size, cudaMemcpyDeviceToHost));
    printVecInVec(h_prompt_tokens, batch_size, max_prompt_len, batch_size, max_prompt_len, "h_prompt_tokens");


    decoding_params.cublas_handle = cublasHandle;
    decoding_params.stream = stream;
    decoding_params.embedding_table = d_embedding_table;
    decoding_params.freq_cis = d_freq_cis;
    decoding_params.prompt_sequence_length = d_prompt_sequence_length;
    decoding_params.prompt_tokens = d_prompt_tokens;
    decoding_params.prompt_tokens_mask = d_prompt_tokens_mask;
    decoding_params.decodingnorm.gamma = d_decoding_resnorm_gamma;
    decoding_params.output_weight.kernel = d_output_weight_kernel;
    decoding_params.output_ids = d_output_ids;
    decoding_params.sequence_length = d_sequence_lengths;
    decoding_params.min_prompt_seq_len = min_prompt_seq_len;
    decoding_params.max_prompt_seq_len = max_prompt_seq_len;

    const tinycudallama::OperationType type = sizeof(T) == sizeof(float) ? tinycudallama::OperationType::FP32 : tinycudallama::OperationType::FP16;

    tinycudallama::DecodingSampling<type> *decoding = new tinycudallama::DecodingSampling<type>(allocator, batch_size, max_prompt_len,
                                                                                                max_gen_len, head_num, size_per_head,
                                                                                                vocab_size, decoder_layers,
                                                                                                end_id, ffn_hidden_units, candidate_num,
                                                                                                probability_threshold);

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    cudaEventQuery(start);

    decoding->forward(param, decoding_params);

    cudaDeviceSynchronize();

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    printf("Time = %g ms.\n", elapsedTime);
    printf("[INFO] batch_size %d topk %d topp %f head_num %d size_per_head %d max_prompt_len %d max_gen_len %d decoder_layers"
           " %d vocab_size %d FL-CPP-decoding-sampling-time %.2f ms\n",
           batch_size, candidate_num, probability_threshold, head_num, size_per_head, max_prompt_len, max_gen_len, decoder_layers,
           vocab_size, elapsedTime);

    int *h_word_ids = new int[batch_size * total_len];
    CHECK_CUDA_ERROR(cudaMemcpy(h_word_ids, decoding_params.output_ids, sizeof(int) * batch_size * total_len, cudaMemcpyDeviceToHost));

    int *h_seq_lengths = new int[batch_size];
    CHECK_CUDA_ERROR(cudaMemcpy(h_seq_lengths, d_sequence_lengths, sizeof(int) * batch_size, cudaMemcpyDeviceToHost));

    printVecInVec(h_seq_lengths, 1, batch_size, 1, batch_size, "h_seq_lengths");

    printf("word_ids:\n[\n");
    for (int i=0; i<batch_size; ++i) {
        printf("[");
        for (int j=0; j<h_seq_lengths[i]; ++j) {
            printf("%d\t", h_word_ids[i * total_len + j]);
        }
        printf("]\n");
    }
    printf("]\n");

    printVecInVec(h_prompt_tokens, batch_size, max_prompt_len, batch_size, max_prompt_len, "h_prompt_tokens");

    printVecInVec(h_prompt_tokens_mask, batch_size, max_prompt_len, batch_size, max_prompt_len, "h_prompt_tokens_mask");



    delete[] param;
    delete[] h_prompt_sequence_length;
    delete[] h_prompt_tokens_mask;
    delete decoding;
    return;
}

int main()
{
    srand(0);
    struct cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("Device %s\n", prop.name);

    const int batch_size = 4;
    const int candidate_num = 0;
    const float probability_threshold = 0.8;
    const int head_num = 8;
    const int size_per_head = 128;
    const int vocab_size = 200;
    const int max_prompt_len = 16;
    const int max_gen_len = 16;
    const int decoder_layers = 4;
    const int ffn_hidden_units = 64 * 2;

    decoding_sample<float>(batch_size, candidate_num, probability_threshold, head_num, size_per_head, vocab_size,
                           max_prompt_len, max_gen_len, decoder_layers, ffn_hidden_units);

    return 0;
}
