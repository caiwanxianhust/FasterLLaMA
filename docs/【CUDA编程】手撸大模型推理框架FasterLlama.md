# 【CUDA编程】手撸一个大模型推理框架 FasterLLaMA

**写在前面**：之前笔者写过 4 篇关于 Nvidia 官方项目 Faster Transformer 的源码解读文章，针对每个 Kernel 分析了作者的实现逻辑和意图。考虑到 Faster Transformer 是一个 Encoder-Decoder 架构的推理框架，而目前的一些开源大模型如 GPT、LLaMA 等都是基于 Decoder-only 架构，早在半年以前笔者就想着手撸一个 Decoder-only 架构的大模型推理框架，但是这段时间笔者工作和家庭事情比较多，最近终于抽出一点点时间完成了第一版。本文将对 FasterLLaMA v1.0 版本源码进行介绍，为了便于交流讨论，除公众号：**后来遇见AI** 以外，本文也将在知乎进行发布，欢迎各位读者阅读并给出意见。

## 1 版本发布背景
在 FasterLLaMA v1.0 中，笔者提供了一个 Decoder 模块和一套推理方案 Decoding 模型，目前 FasterLLaMA v1.0 仅适配 LLaMA2，至于LLaMA3 及其他开源大模型的适配工作，将在后续版本逐步加入。其中，Decoder 相当于我们常说的 decoder layer；而 Decoding 则包含了整个解码的流程，包括词嵌入、解码层和采样解码等过程。

针对 Decoder 模块的 GEMM 场景，笔者提供了基于 cuBLAS 的 INT8 量化实现，对模型权重和激活值进行 INT8 量化，量化粒度均为 per-channel，通过 INT8 量化的矩阵运算可以高效地利用 GPU 中的 INT8 Tensor Core，在保证低精度损失的前提下，取得较好的加速比（对比 FP16 运算精度而言），要注意的是 FasterLLaMA v1.0 仅支持在计算能力不低于 7.5 的设备上运行。另外，`Q*K` 乘法和 `QK*V` 乘法部分在 v1.0 版本仍然还是使用的 FP32 类型，没有实现低精度量化。

针对 Decoding 模型的解码场景，笔者参考了 Faster Transformer，提供了两种基于采样解码的实现：top-k 解码和 top-p 解码。

数据类型方面，目前 FasterLLaMA v1.0 支持 FP32 和 FP16 两种类型，笔者针对 FP16 类型对相关 Kernel 函数模板进行了特化。

注意力机制方面，目前 FasterLLaMA v1.0 仅支持 MHA，计划在后续版本加入对 MQA 和 GQA 的支持。

## 2 整体架构
FasterLLaMA v1.0 基于 CUDA、cuBLAS、CUB 等 Nvidia 官方库实现，目前仅提供 C++ API，用户可以将它集成到本机 C++ 中构建的推理服务代码中。此外笔者还提供了一些简单的示例代码来演示如何在 C++ 中执行 Decoding 过程。

下面是 Decoder 模块的整体架构图：
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qX4u3gKYjsOZ7r3ib6Jk02RkszQibYbxMpzTOPryIsOxonbFgQicponrNVqWCrIvZiasb0heJcevSic3g/640?wx_fmt=png&amp;from=appmsg)

下面是 Decoding 模型的整体架构图：
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qX4u3gKYjsOZ7r3ib6Jk02RdlQvOO3VpWo7Z3icRXiat9UOE6lAkwyiaETUsP34h7DGFgZ3s6NLtrfvQ/640?wx_fmt=png&amp;from=appmsg)

源码地址如下，有兴趣的读者可以前往下载：
> https://github.com/caiwanxianhust/FasterLLaMA

## 3 Decoder 模块
### 3.1 OpenDecoder 类结构
了解大模型训练推理过程的读者都知道，大模型的推理可以分为 prompt 和 generation 两个阶段，两个阶段在处理时的差异在于数据维度的差异，即 prompt 阶段是多 token 输入，即 from_tensor 维度为 `[batch_size, seq_len, hidden_units]`，而 generation 阶段的输入则是单 token，即 from_tensor 维度为 `[batch_size, 1, hidden_units]`，所以前者更多的计算算子是 gemm，而后者更多的计算算子则是 gemv，在 `OpenDecoder` 类模版中笔者提供的 Kernel 都兼容两个阶段的计算，无论哪个阶段都直接执行 forward 函数即可，具体参见 forward 函数的定义。

`OpenDecoder` 类模版有两个模版参数：`OpType_` 指明了激活值的数据类型，通常就是 FP32、FP16，`QuantizationType` 指明了量化程度，这里就是 INT8。

除了一些 Transformer 结构相关的成员变量以外，主要就是声明了几个变量 buffer，具体用处和含义如下：
- `from_tensor_int8_buf_`：存储 `int8_t` 类型的激活值矩阵，这个 buffer 会在不同的环节复用以节省设备内存。
- `from_tensor_scale_buf_`：存储 `int8_t` 类型的激活值矩阵对应的缩放比例，主要用于反量化，同样地，这个 buffer 会在不同的环节复用以节省设备内存。
- `query_buf_`：存储 `int32_t` 类型的 query 矩阵，这个 buffer 会在不同的环节复用以节省设备内存，比如后面 FFN 结构中的 w1 gemm 的输出矩阵也缓存在这个 buffer。
- `key_buf_`：存储 `int32_t` 类型的 key 矩阵，这个 buffer 会在不同的环节复用以节省设备内存，比如后面 FFN 结构中的 w3 gemm 的输出矩阵也缓存在这个 buffer。
- `value_buf_`：存储 `int32_t` 类型的 value 矩阵，这个 buffer 会在不同的环节复用以节省设备内存，比如后面 FFN 结构中的 w2 gemm 的输出矩阵也缓存在这个 buffer。
- `query_out_buf_`：存储 `float` 类型的 query 矩阵。
- `key_out_buf_`：存储 `float` 类型的 key 矩阵。
- `value_out_fp_buf_`：存储 `float` 类型的 value 矩阵。
- `qk_buf_`：存储 `float` 类型的 qk 矩阵。
- `qkv_buf_`：存储 `float` 类型的 qkv 矩阵。
- `ffn_tensor_buf_`：存储 `DataType` 类型的 FFN 结构的输入 tensor，用于后面的残差结构。
- `ffn_inter_scale_buf_`：存储 FFN 结构中 w2 gemm 前的量化缩放比例。
```cpp
template <OperationType OpType_, OperationType QuantizationType>
class OpenDecoder
{
private:
    typedef DecoderTransformerTraits<OpType_> Traits_;
    typedef DecoderTransformerTraits<OperationType::FP32> qkv_Traits_;
    typedef DecoderTransformerTraits<QuantizationType> weight_Traits_;

    typedef typename Traits_::DataType DataType_;
    typedef typename weight_Traits_::DataType weight_DataType_;
    DecoderInitParam<DataType_, weight_DataType_> param_;

    int cublasAlgo_[5];

    int batch_size_;
    int max_prompt_len_;
    int max_gen_len_;
    int total_len_;
    int head_num_;
    int size_per_head_;
    int hidden_units_;

    /*  buf_size = batch_size * max_prompt_len_ * head_num * size_per_head
        cache_size = batch_size * head_num * total_len_ * size_per_head
    */
    int8_t *from_tensor_int8_buf_; // buf_size, [batch_size * seq_len, head_num * size_per_head]
    float *from_tensor_scale_buf_; // batch_size * max_prompt_len_, [batch_size, seq_len]
    int32_t *query_buf_;           // buf_size, [batch_size * seq_len, head_num * size_per_head]
    int32_t *key_buf_;             // buf_size, [batch_size * seq_len, head_num * size_per_head]
    int32_t *value_buf_;           // buf_size, [batch_size * seq_len, head_num * size_per_head]
    float *query_out_buf_;         // buf_size, [batch_size, head_num, seq_len, size_per_head]
    float *key_out_buf_;           // buf_size, [batch_size, head_num, seq_len, size_per_head]
    float *value_out_fp_buf_;      // buf_size, [batch_size, head_num, seq_len, size_per_head]
    float *qk_buf_;                // [batch_size * head_num, seq_len, total_len_]
    float *qkv_buf_;               // buf_size, [batch_size * head_num, seq_len, size_per_head]
    DataType_ *ffn_tensor_buf_;    // buf_size, [batch_size, seq_len, head_num * size_per_head]
    float *ffn_inter_scale_buf_;   // batch_size * max_prompt_len_, [batch_size, seq_len]

public:
    OpenDecoder(int batch_size, int max_prompt_len, int max_gen_len, int head_num, int size_per_head);

    void initialize(DecoderInitParam<DataType_, weight_DataType_> param, char *buf);

    int getWorkspaceSize();

    /**
        * key_cache_ value_cache_: cache_size, [batch_size, head_num, total_len_, size_per_head]
        * freq_cis_: [max_prompt_len_, size_per_head]
        */
    void forward(const DataType_ *from_tensor, const float *freq_cis, float *key_cache_, float *value_cache_, int ffn_hidden_units,
                    DataType_ *decoder_output, const int start_pos, const int seq_len);

    ~OpenDecoder();
};
```

除了一些成员变量以外，`OpenDecoder` 类模版还提供了 3 个主要的成员方法：`initialize` 方法在 `forward` 方法执行前执行，用于划分各 buffer 的地址空间；`getWorkspaceSize` 在实例化之后执行，用于计算所有的 buffer 所需的设备内存大小；`forward` 方法就是 decoder layer 的计算过程。

### 3.2 ResNormQuantizedKernel
从 Decoder 架构示意图中可以看出，第一步首先进行 ResNorm（即 Root Mean Square Layer Normalization，均方根归一化），通常 Norm 的目的是想通过缩放和偏移使模型在激活值和权重上出现噪声时依旧具有稳定的分布，同时又能很好的控制激活值和权重的取值范围。例如比较经典的 Layer Normalization（LN，层归一化）和 Batch Normalization（BN，批归一化），前者在 NLP 中运用较多，后者在 CV 任务中则有更多的应用，而 LLaMA 中的 ResNorm 则类似 LN 的变种，在归一化的过程中移除了 LN 中的平移操作（即去掉了均值的计算和减除步骤），而只保留了缩放操作。因此，ResNorm 仅依赖于输入特征的均方根（Root Mean Square）来进行归一化，ResNorm 的计算公式如下：
$$
ResNorm(x) = \gamma \frac{x}{\sqrt{\frac{1}{n} \sum _{i=1}^{n} x_i^2 + \epsilon}}
$$

对 `from_tensor` 进行 ResNorm 之后，紧接着还有一个 INT8 量化操作，将后面 Q\K\V 矩阵乘法的输入矩阵数据类型转换为 `int8_t` 类型，为了减少 Kernel 启动开销，笔者将 ResNorm 和 INT8 量化融合到一个 Kernel 中，即 `resNormQuantizedKernel`，下面来看一下计算代码：
```cpp
/** resNorm、量化
* grid(batch_size * seq_len)  block(256)
* output: [batch_size, seq_len, hidden_units]
* input: [batch_size, seq_len, hidden_units]
* gamma: [hidden_units, ]
*/
template <typename DataType>
__global__ void resNormQuantizedKernel(int8_t *__restrict__ output, const DataType *__restrict__ input, const DataType *__restrict__ gamma,
                                        float *__restrict__ norm_scale, const float eps, const int hidden_units)
{
    const int row_id = blockIdx.x;
    const int offset = row_id * hidden_units;

    extern __shared__ float s_buf[]; // hiddent_units
    float val;
    float mean = 0.0f;
    float absmax = -1e9f;
    char4 out_val;

    for (int tid = threadIdx.x; tid < hidden_units; tid += blockDim.x)
    {
        val = static_cast<float>(input[offset + tid]);
        s_buf[tid] = val;
        mean += val * val;
        absmax = max(absmax, fabsf(val * static_cast<float>(__ldg(gamma + tid))));
    }
    __syncthreads();

    mean = blockAllReduceSum<float>(mean);
    mean = rsqrtf(mean / hidden_units + eps);

    absmax = blockAllReduceMax<float>(absmax);
    absmax *= mean;
    if (threadIdx.x == 0)
    {
        norm_scale[blockIdx.x] = absmax / 127.0f;
    }

    int target_idx;
    char4 *out_ptr = (char4 *)output;
    for (int tid = (threadIdx.x << 2); tid < hidden_units; tid += (blockDim.x << 2))
    {
        out_val.x = float_to_int8_rn(s_buf[tid] * mean * static_cast<float>(__ldg(gamma + tid)) * 127.0f / absmax);
        out_val.y = float_to_int8_rn(s_buf[tid + 1] * mean * static_cast<float>(__ldg(gamma + tid + 1)) * 127.0f / absmax);
        out_val.z = float_to_int8_rn(s_buf[tid + 2] * mean * static_cast<float>(__ldg(gamma + tid + 2)) * 127.0f / absmax);
        out_val.w = float_to_int8_rn(s_buf[tid + 3] * mean * static_cast<float>(__ldg(gamma + tid + 3)) * 127.0f / absmax);
        target_idx = row_id * hidden_units + tid;
        out_ptr[target_idx >> 2] = out_val;
    }
}

template <>
__global__ void resNormQuantizedKernel(int8_t *__restrict__ output, const half *__restrict__ input, const half *__restrict__ gamma,
                                        float *__restrict__ norm_scale, const float eps, const int hidden_units)
{
    const int row_id = blockIdx.x;
    const int offset = row_id * hidden_units;

    extern __shared__ float s_buf[]; // hiddent_units
    float val;
    float mean = 0.0f;
    float absmax = -1e9f;
    char4 out_val;

    for (int tid = threadIdx.x; tid < hidden_units; tid += blockDim.x)
    {
        val = static_cast<float>(input[offset + tid]);
        s_buf[tid] = val;
        mean += val * val;
        absmax = max(absmax, fabsf(val * __half2float(__ldg(gamma + tid))));
    }
    __syncthreads();

    mean = blockAllReduceSum<float>(mean);
    mean = rsqrtf(mean / hidden_units + eps);

    absmax = blockAllReduceMax<float>(absmax);
    absmax *= mean;
    if (threadIdx.x == 0)
    {
        norm_scale[blockIdx.x] = absmax / 127.0f;
    }

    int target_idx;
    char4 *out_ptr = (char4 *)output;
    for (int tid = (threadIdx.x << 2); tid < hidden_units; tid += (blockDim.x << 2))
    {
        out_val.x = float_to_int8_rn(s_buf[tid] * mean * __half2float(__ldg(gamma + tid)) * 127.0f / absmax);
        out_val.y = float_to_int8_rn(s_buf[tid + 1] * mean * __half2float(__ldg(gamma + tid + 1)) * 127.0f / absmax);
        out_val.z = float_to_int8_rn(s_buf[tid + 2] * mean * __half2float(__ldg(gamma + tid + 2)) * 127.0f / absmax);
        out_val.w = float_to_int8_rn(s_buf[tid + 3] * mean * __half2float(__ldg(gamma + tid + 3)) * 127.0f / absmax);
        target_idx = row_id * hidden_units + tid;
        out_ptr[target_idx >> 2] = out_val;
    }
}

template <typename DataType>
void launchResNormQuantizedKernel(int8_t *output, const DataType *input, const DataType *gamma,
                                    float *norm_scale, const float eps, const int nrows, const int hidden_units, cudaStream_t stream)
{
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    assert(hidden_units % 4 == 0);
    int mem_size = sizeof(float) * hidden_units;
    resNormQuantizedKernel<DataType><<<nrows, 256, mem_size, stream>>>(output, input, gamma, norm_scale, eps, hidden_units);
}
```

`input` 矩阵的形状为 `[batch_size, seq_len, hidden_units]`，无论是 ResNorm 还是 INT8 量化都是在 `hidden_units` 维度上进行计算，所以核函数的 `grid_size` 直接取 `batch_size * seq_len`，而 `block_size` 这里直接取 `256`，至于为什么选择 `256` 而不是其他的值，主要是综合考虑到 SM 占用率和寄存器使用量的一个经验值，笔者在其他文章中有过详细分析，这里不再赘述，通常情况下，`block_size` 取 `128` 或者 `256` 都会有比较好的性能表现。

由于 `hidden_units` 的值通常大于 `block_size`，所以每个线程要处理多个元素，也就意味着对于某个线程来说无法把每次访问 `input` 读取的 `val` 临时放进寄存器暂存以备下次取用，为了减少多次读取全局内存带来的内存访问开销，笔者在这里定义了一个共享内存变量 `s_buf`，一个长度为 `hidden_units` 的数组，用来临时存储 `hidden_units` 维度上的每个 `val`。当然，这里函数签名中对 `input` 参数进行了 `const T *__restrict__` 修饰，在 Maxwell 以上的架构中，CUDA 编译器会视情况利用 Unified Cache 进行缓存优化，达到纹理内存的加速效果，本身速度也不慢，所以这里任选一种加速方式均可。

从第一次遍历 `input` 时可以发现笔者执行了两项任务一个是求 `mean`，一个是求 `absmax`。前者是为了计算 ResNorm 中的 RMS 值，也就是均方根的值，所以先对每个线程内的 `val` 值求平方和，然后在 block 内规约求和，就得到了 `hidden_units` 个 `val` 的平方和，然后加上 $\epsilon$ 后求均方根即可；后者是为了 INT8 量化，关于 INT8 量化的意义笔者在前面的文章中进行过详细介绍，这里不在赘述，量化方法很简单，求绝对值的最大值 `absmax`，然后把最大值量化到 `int8_t` 类型的最大值 `127` 即可，根据计算公式，由于 RMS 的值对于每个分量都是一样的，所以只需要找到 $\gamma \cdot x$ 的最大值就可以了，思路跟求 `mean` 一样，先是线程内循环比较，然后 block 内规约求最大值，得到最大值后再乘以 `mean` 就是最终输出分量的最大值，把缩放比例 `absmax / 127.0f` 存入 `norm_scale`，以备后面反量化的时候使用。RMS、缩放比例均已知的情况下，计算最终的输出分量就直接按照公式即可，从 `s_buf` 中取出输入 `val` 值，乘以 $\gamma$，除以 RMS，再除以缩放比例即可。

另外，在 kernel 中为了加速内存写入，笔者使用了 CUDA C++ 内置向量类型 `float4` 和 `char4`，一次性写入相邻的 4 个 `float` 和 `int8_t` 元素，这是一种常用的优化技巧，目的是为了提升读写带宽，类似的，笔者在之前的文章中也有大量介绍，这里就不再详细说明了。

### 3.3 Q\K\V GEMM
根据 from_tensor 生成 Q\K\V，无论是在 prompt 阶段还是 generation 阶段都是 GEMM 而不是 GEMV，因为 from_tensor 的维度为 `[batch_size, seq_len, hidden_units]`，即使 `seq_len` 为 `1`，也仍然是矩阵乘法操作。这三个 GEMM 操作笔者直接调用了 cuBLAS 库的 GEMM API 进行，其中权重矩阵都是推理前提前训练好并且量化后的 INT8 矩阵，数据类型为 INT8 * INT8 -> INT32，由于是直接调用 API，所以不再赘述，直接读源码即可。

### 3.4 QKRoteEmbeddingTranspose
生成 Q\K\V 之后，对于 Q\K 需要进行旋转位置编码（Rope）、分头（transpose）操作，得到形状为 `[batch_size, head_num, seq_len, size_per_head]` 的 Q\K 矩阵，而对于 V 无需 Rope 只需要分头即可，此外由于上一步矩阵乘法得到的是未反量化的 Q\K\V 矩阵，所以这里还隐含了一个反量化的操作。在这里笔者把 Q\K 的三项操作融合到了一个 Kernel 里完成，即 `QKRoteEmbeddingTranspose`，针对不同的 `size_per_head` 大小笔者提供了 3 个 Kernel，这里篇幅有限，就以 `size_per_head` 等于 `128` 时的 `warpQKRoteEmbeddingTransposeKernel` 为例进行介绍。

在此之前，需要介绍一下旋转位置编码 Rope 的计算思想，关于为什么要使用旋转位置编码网上也有大量的文章介绍，这里咱们只关注如何进行计算。根据 LLaMA2 源码，要计算 Rope，首先要计算一个 freqs_cis 矩阵，在计算之前我们先来看一下 Rope 的计算公式：

$$
R_{\theta, m}^{d} x=\left(\begin{array}{r}
x_{1} \\
x_{2} \\
x_{3} \\
x_{4} \\
\ldots \\
x_{d-1} \\
x_{d}
\end{array}\right) \otimes\left(\begin{array}{r}
\cos \left(m \theta_{1}\right) \\
\cos \left(m \theta_{1}\right) \\
\cos \left(m \theta_{2}\right) \\
\cos \left(m \theta_{2}\right) \\
\ldots \\
\cos \left(m \theta_{d / 2}\right) \\
\cos \left(m \theta_{d / 2}\right)
\end{array}\right)+\left(\begin{array}{r}
-x_{2} \\
x_{1} \\
-x_{4} \\
x_{3} \\
\ldots \\
-x_{d} \\
x_{d-1}
\end{array}\right) \otimes\left(\begin{array}{r}
\sin \left(m \theta_{1}\right) \\
\sin \left(m \theta_{1}\right) \\
\sin \left(m \theta_{2}\right) \\
\sin \left(m \theta_{2}\right) \\
\ldots \\
\sin \left(m \theta_{d / 2}\right) \\
\sin \left(m \theta_{d / 2}\right)
\end{array}\right)
$$

其中，每组旋转角度 $\theta$ 的计算方式如下，这也是一个经验公式：
$$
\theta _j = 10000 ^{-2j/d} ， j \in [1, 2, \cdots, d/2]
$$

从计算公式可以看出，抛开变量 $x$，Rope 就只跟 $m$ 和 $d$ 两个变量有关，分别对应 `seq_len` 和 `size_per_head`，也就是说，只要 `size_per_head` 没有变化，freqs_cis 矩阵就是确定的，完全可以预先计算一个长度为最大可支持长度的 freqs_cis 矩阵以备后续计算随时读取。下面我们来看一下 freqs_cis 矩阵的生成方式，笔者这里针对 `size_per_head` 长度不同分别给了两个 Kernel，下面来看一下代码。

```cpp
/** precomputeFreqsCis
* grid(seq_len)  block(block_size) for size_per_head/2 >= block_size(128)
* freq_cis: [seq_len, size_per_head]
*/
__global__ void precomputeFreqsCis(float *freq_cis, const int size_per_head)
{
    int offset = blockIdx.x * size_per_head;
    for (int i = threadIdx.x; i < (size_per_head >> 1); i += blockDim.x)
    {
        float val = i * (-2.0f) / size_per_head;
        float theta = __powf(1e4f, val) * blockIdx.x;
        freq_cis[offset + 2 * i] = __cosf(theta);
        freq_cis[offset + 2 * i + 1] = __sinf(theta);
    }
}

/**
    * block(32, 4)   each warp compute one row
    */
__global__ void warpPrecomputeFreqsCis(float *freq_cis, const int size_per_head, const int seq_len)
{
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    int offset = row * size_per_head;
    if (row < seq_len)
    {
        for (int i = threadIdx.x; i < (size_per_head >> 1); i += blockDim.x)
        {
            float val = i * (-2.0f) / size_per_head;
            float theta = __powf(1e4f, val) * row;
            freq_cis[offset + 2 * i] = __cosf(theta);
            freq_cis[offset + 2 * i + 1] = __sinf(theta);
        }
    }
}

void launchPrecomputeFreqsCis(float *freq_cis, const int size_per_head, const int seq_len, cudaStream_t stream)
{
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    if ((size_per_head / 2) < 128)
    {
        int warp_num = 128 / 32;
        int grid_size = (seq_len + warp_num - 1) / warp_num;
        dim3 grid(grid_size);
        dim3 block(32, warp_num);
        warpPrecomputeFreqsCis<<<grid, block, 0, stream>>>(freq_cis, size_per_head, seq_len);
    }
    else
    {
        dim3 grid(seq_len);
        dim3 block(128);
        precomputeFreqsCis<<<grid, block, 0, stream>>>(freq_cis, size_per_head);
    }
}
```

从代码逻辑可以看出，`warpPrecomputeFreqsCis` 顾名思义就是一个 warp 完成一行（`size_per_head` 个元素）计算任务，最后得到 freqs_cis 矩阵每一行的元素分别为：
$$
[cos(m \theta _1), sin(m \theta _1), \cdots, cos(m \theta _{d/2}), sin(m \theta _{d/2})]
$$

计算逻辑完全按照公式进行，没有使用什么优化技巧。

再来看一下 `warpQKRoteEmbeddingTransposeKernel` 核函数，总共完成反量化、Rope、转置三个操作，由于 Q\K 的操作步骤是一致的，所以可以在一个 Kernel 中完成，可以通过将其放入不同的 block 进行计算。首先定义 block 为 `(32, 4)`，也就是说每个 warp（32 个线程）完成一行元素的计算任务，一个 block 内完成 4 个 head 的计算任务。grid 定义为 `(head_num / 4, seq_len, batch_size * 2)`，即通过 `blockIdx.z` 区分当前 block 计算的是 Q 还是 K。

根据内置变量 `blockIdx` 和 `threadidx` 定义矩阵的各维度索引，然后根据索引定位当前线程要计算的元素位置。为了提升内存写入带宽，利用内置向量类型 `float4` 一次写入 4 个元素，将输出指针强转为 `float4 *` 类型的 `out_ptr`。随后进行各任务的计算，首先是反量化，需要将输入值分别乘以激活矩阵对应的缩放比例 `inp_scale_val`、权重矩阵对应的缩放比例 `weight_scale`，这里要特别注意的是，对于激活矩阵，其量化的维度是 -1 维，也就是 `hidden_units` 所在的维度，而权重矩阵其量化的维度是第 0 维，这是矩阵运算的特殊性决定的，所以 `weight_scale` 取值时要注意索引，这里通过 `__ldg()` 取值可以利用 L1 缓存加速读操作。然后是 Rope 计算，根据计算公式一次完成 4 个分量的计算，两两一组。最后进行分头操作，也就是将 `[batch_size, seq_len, head_num, size_per_head]` 转置为 `batch_size, head_num, seq_len, size_per_head`。

```cpp
/**
* 反量化、rope旋转编码、转置
* Q K: [batch_size, seq_len, head_num, size_per_head]
* grid(head_num / warp_num, seq_len, batch_size * 2) block(32, warp_num), each warp process size_per_head elements
* q_inp_sacle k_inp_scale: [batch_size, seq_len], absmax / 127.0f
* q_weight_scale k_weight_scale: [head_num * size_per_head, ], absmax / 127.0f
* freq_cis: [max_seq_len, size_per_head]
*/
__global__ void warpQKRoteEmbeddingTransposeKernel(float *q_buf, float *k_buf, const int32_t *Q,
                                                    const int32_t *K, const float *q_inp_scale, const float *k_inp_scale,
                                                    const float *q_weight_scale, const float *k_weight_scale, const float *freq_cis,
                                                    const int batch_size, const int seq_len,
                                                    const int start_pos, const int total_len, const int head_num,
                                                    const int size_per_head)
{
    const int qk_id = blockIdx.z / batch_size;
    const int batch_id = blockIdx.z % batch_size;
    const int seq_id = blockIdx.y;
    const int head_id = blockIdx.x * blockDim.y + threadIdx.y;
    const int offset = batch_id * seq_len * head_num * size_per_head + seq_id * head_num * size_per_head + head_id * size_per_head;
    const float inp_scale_val = (qk_id == 0) ? __ldg(q_inp_scale + batch_id * seq_len + seq_id) : __ldg(k_inp_scale + batch_id * seq_len + seq_id);
    const int32_t *data_ptr = (qk_id == 0) ? Q + offset : K + offset;
    const float *weight_scale_ptr = (qk_id == 0) ? q_weight_scale : k_weight_scale;
    const float *freq_cis_ptr = freq_cis + seq_id * size_per_head;

    float4 *out_ptr = (qk_id == 0) ? (float4 *)q_buf : (float4 *)k_buf;
    float4 val, rope_val;
    const int tid = (threadIdx.x << 2);
    int target_idx;
    if (tid < size_per_head)
    {
        // dequantized
        val.x = static_cast<float>(data_ptr[tid]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid);
        val.y = static_cast<float>(data_ptr[tid + 1]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid + 1);
        val.z = static_cast<float>(data_ptr[tid + 2]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid + 2);
        val.w = static_cast<float>(data_ptr[tid + 3]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid + 3);

        // rope embedding
        rope_val.x = val.x * freq_cis_ptr[tid] - val.y * freq_cis_ptr[tid + 1];
        rope_val.y = val.y * freq_cis_ptr[tid] + val.x * freq_cis_ptr[tid + 1];
        rope_val.z = val.z * freq_cis_ptr[tid + 2] - val.w * freq_cis_ptr[tid + 3];
        rope_val.w = val.w * freq_cis_ptr[tid + 2] + val.z * freq_cis_ptr[tid + 3];

        // transpose
        target_idx = batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head + seq_id * size_per_head + tid;
        out_ptr[target_idx >> 2] = rope_val;
    }
}
```

### 3.5 DequantizedVTransposeKernel
前面介绍过，生成 Q\K\V 之后，对于 Q\K 需要进行旋转位置编码（Rope）、分头（transpose）操作，而对于 V 只需要分头即可，另外再加一个反量化操作，笔者把这两个操作都融合到一个 Kernel 里，根据 `size_per_head` 的不同笔者提供了 3 个 Kernel，这里以 `warpDequantizedVTransposeKernel` 为例介绍一下 cuda 实现逻辑。

```cpp
/**
* 反量化、转置
* grid(head_num / warp_num, seq_len, batch_size) block(32, warp_num), each warp process size_per_head elements
* V: [batch_size, seq_len, head_num, size_per_head]
* v_buf: [batch_size, head_num, seq_len, size_per_head]
* v_inp_sacle: [batch_size, seq_len], absmax / 127.0f
* v_weight_scale: [head_num * size_per_head, ], absmax / 127.0f
*/
__global__ void warpDequantizedVTransposeKernel(float *__restrict__ v_buf, const int32_t *__restrict__ V,
                                                const float *__restrict__ v_inp_scale, const float *__restrict__ v_weight_scale,
                                                const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    const int batch_id = blockIdx.z;
    const int seq_id = blockIdx.y;
    const int head_id = blockIdx.x * blockDim.y + threadIdx.y;
    const int offset = batch_id * seq_len * head_num * size_per_head + seq_id * head_num * size_per_head + head_id * size_per_head;
    const float inp_scale_val = __ldg(v_inp_scale + batch_id * seq_len + seq_id);
    const int32_t *data_ptr = V + offset;
    float4 *out_ptr = (float4 *)v_buf;
    float4 val;
    const int tid = (threadIdx.x << 2);
    int target_idx;
    if (tid < size_per_head)
    {
        // dequantized
        val.x = static_cast<float>(data_ptr[tid]) * inp_scale_val * __ldg(v_weight_scale + head_id * size_per_head + tid);
        val.y = static_cast<float>(data_ptr[tid + 1]) * inp_scale_val * __ldg(v_weight_scale + head_id * size_per_head + tid + 1);
        val.z = static_cast<float>(data_ptr[tid + 2]) * inp_scale_val * __ldg(v_weight_scale + head_id * size_per_head + tid + 2);
        val.w = static_cast<float>(data_ptr[tid + 3]) * inp_scale_val * __ldg(v_weight_scale + head_id * size_per_head + tid + 3);

        // transpose
        target_idx = batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head + seq_id * size_per_head + tid;
        out_ptr[target_idx >> 2] = val;
    }
}
```
计算思路与前面 `warpQKRoteEmbeddingTransposeKernel` 基本相同只是去掉了 Rope 操作。首先定义 block 为 `(32, 4)`，也就是说每个 warp（32 个线程）完成一行元素的计算任务，一个 block 内完成 4 个 head 的计算任务。grid 定义为 `(head_num / 4, seq_len, batch_size)`。

为了提升内存写入带宽，同样也利用了内置向量类型 `float4` 一次写入 4 个元素，将输出指针强转为 `float4 *` 类型的 `out_ptr`。随后进行各任务的计算，首先是反量化，将输入值分别乘以激活矩阵对应的缩放比例 `inp_scale_val`、权重矩阵对应的缩放比例 `weight_scale`，通过 `__ldg()` 取值利用 L1 缓存加速读操作。然后进行分头操作，将 `[batch_size, seq_len, head_num, size_per_head]` 转置为 `batch_size, head_num, seq_len, size_per_head`。

### 3.6 StoreKVcacheKernel
我们知道 attention 中 softmax 计算的对象是 Q 和 K 的乘积，query 我们已经拿到了，就是当前解码 step 的输入 tensor 变换后的结果。K 是什么？对于当前 step 的 Q 来说这里的 K 应该是前面 step 的 token 对应的 tensor 变换后的结果，由于生成 K 的 Dense 变换的权重是固定的且 token 也是确定的，所以 K 也是固定的，那么我们每轮 step 的时候就可以计算好当前 step 的 K 存入 `key_cache` 中供后面的 step 计算时使用，同时在当前 step 也可以从 `key_cache` 中取前面 step 的 K 用于计算。这就是所谓的 KV cache 机制，这样做的好处是什么？每次计算 Q*K 乘法的时候直接取前面 step 计算好的 K 就行了，无需重复计算，以空间换时间。

那么现在我们 Q\K\V 生成并且旋转编码、分头之后就需要把当前 step 的 token 对应的 K\V 存入 KV cache 以备后面的 step 计算使用。这里笔者让 `key_cache` 和 `value_cache` 按 `[batch_size, head_num, total_len_, size_per_head]` 的内存顺序存储，也就是说对于每个 step 来说存入的 `key_cache` 和 `value_cache` 的数据不是连续存储的。这样做的好处是什么？可以减少 `key_cache` 和 `value_cache` 中数据的移动次数，后面如果要计算 Q*K，可以直接拿 `key_cache` 作为 K 矩阵参与运算，设置好主维度参数即可。具体实现代码如下：

```cpp
/**
* grid: [seq_len, head_num / blockDim.y, batch_size * 2]  block(size_per_head / 4, 256 / (size_per_head / 4))
* k_cache v_cache: [batch_size, head_num, max_seq_len, size_per_head]
* K V : [batch_size, head_num, seq_len, size_per_head]
*/
__global__ void storeKVcacheKernel(float *__restrict__ k_cache, float *__restrict__ v_cache, const float *__restrict__ K,
                                    const float *__restrict__ V, const int start_pos, const int seq_len, const int batch_size, const int head_num,
                                    const int max_seq_len, const int size_per_head)
{
    const int kv_id = blockIdx.z / batch_size;
    const int batch_id = blockIdx.z % batch_size;
    const int head_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int seq_id = blockIdx.x;
    const int offset = batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head + seq_id * size_per_head;
    const int cache_offset = batch_id * head_num * max_seq_len * size_per_head + head_id * max_seq_len * size_per_head +
                                (start_pos + seq_id) * size_per_head;
    float4 *cache_ptr = (kv_id == 0) ? (float4 *)(k_cache + cache_offset) : (float4 *)(v_cache + cache_offset);
    const float4 *data_ptr = (kv_id == 0) ? (const float4 *)(K + offset) : (const float4 *)(V + offset);
    cache_ptr[threadIdx.x] = data_ptr[threadIdx.x];
}
```
这里还是以 `size_per_head` 为 128 时的 `storeKVcacheKernel` 为例，设置 grid 为 `(seq_len, head_num / 8, batch_size * 2)`，block 为 `(32, 8)`。每个 warp 处理一行元素，每个 block 处理 8 个 head，通过 `blockIdx.z` 区分当前 block 计算的是 K 还是 V。

为了提高内存读写带宽，这里也是利用了内置向量类型 `float4` 一次读写 4 个元素，常规加速技巧，不再赘述，目标索引就是按照前面说的内存排列顺序计算即可。

### 3.7 Q*K 乘法
根据 Attention 计算公式，Q\K 生成之后就开始计算 `Q*K`，这里 Q 的形状为 `[batch_size, head_num, seq_len, size_per_head]`，在 2、3 两个维度上进行乘法运算，可以看出不同 decoding 阶段的乘法形式有所不同，在 generation 阶段，`seq_len` 的值为 `1`，所以 `Q*K` 实际上是一个 GEMV 运算，而在 prompt 阶段，`Q*K` 是一个 GEMM 运算，这里笔者进行了一个简单判断，分别调用不同的 cuBLAS API。

在 prompt 阶段，Q\K 直接就使用前面 `QKRoteEmbeddingTranspose` 的计算结果 `query_out_buf_` 和 `key_out_buf_` 进行矩阵运算即可，参数设置比较常规，具体参见源码。

在 generation 阶段，GEMV 的 M 就是 `key_cache`，V 则是 `query_out_buf_`，调用 `cublasSgemvStridedBatched` API 传参计算即可。关于 `key_cache` 中参与计算的行数，笔者这里取了 `start_pos + seq_len`，也就是说把当前 step 的 token 计算的 K 也算进去了，实际上当前 token 的 K 是不应该参与计算的，不过没关系，在后面计算 softmax 时会被 mask 掉。

### 3.8 Softmax
根据 Attention 计算公式，`Q*K` 计算完成之后需要进行 softmax 计算，这里 QK 的形状为 `[batch_size, head_num, seq_len_q, seq_len_k]`，在 -1 维度上进行 softmax。在 prompt 阶段，`seq_len_q` 与 `seq_len_k` 相同，在 generation 阶段，QK 的形状则变为 `[batch_size, head_num, 1, start_pos + 1]`，笔者提供了一个 softmax Kernel 兼容这两种情况的计算逻辑。要注意的是，`attn_mask` 在传参时的偏移量。

```cpp
/**
* softmax
* grid(seq_len_q, head_num, batch_size), block(128), each block process seq_len_k elements
* qk score: [batch_size, head_num, seq_len_q, seq_len_k]
* atten_mask: [max_seq_len, max_seq_len]
*
*/
__global__ void blockSoftmaxKernel(float *__restrict__ qk, const float *__restrict__ attn_mask, const int batch_size,
                                    const int head_num, const int seq_len_q, const int seq_len_k, const int max_seq_len, const float scaler)
{
    const int batch_id = blockIdx.z;
    const int head_id = blockIdx.y;
    const int seq_q_id = blockIdx.x;
    const int offset = batch_id * head_num * seq_len_q * seq_len_k + head_id * seq_len_q * seq_len_k + seq_q_id * seq_len_k;
    const int mask_offset = seq_q_id * max_seq_len;
    extern __shared__ float s_buf[];
    float val, mask_val;
    float sum_val = 0.0f;
    float max_val = -1e9f;
    for (int i = threadIdx.x; i < seq_len_k; i += blockDim.x)
    {
        mask_val = (attn_mask) ? attn_mask[mask_offset + i] : 0.0f;
        val = qk[offset + i] * scaler + mask_val;
        s_buf[i] = val;
        max_val = max(max_val, val);
    }
    __syncthreads();
    max_val = blockAllReduceMax<float>(max_val);

    for (int i = threadIdx.x; i < seq_len_k; i += blockDim.x)
    {
        val = expf(s_buf[i] - max_val);
        sum_val += val;
        s_buf[i] = val;
    }
    __syncthreads();
    sum_val = blockAllReduceSum<float>(sum_val);

    for (int i = threadIdx.x; i < seq_len_k; i += blockDim.x)
    {
        qk[offset + i] = s_buf[i] / sum_val;
    }
}

void launchBlockSoftmaxKernel(float *qk, const float *attn_mask, const int batch_size, const int head_num, const int seq_len_q,
                                const int seq_len_k, const int max_seq_len, const float scaler, cudaStream_t stream)
{
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    dim3 grid(seq_len_q, head_num, batch_size);
    dim3 block(128);
    int shared_mem_size = sizeof(float) * seq_len_k;
    blockSoftmaxKernel<<<grid, block, shared_mem_size, stream>>>(qk, attn_mask, batch_size, head_num, seq_len_q, seq_len_k,
                                                                    max_seq_len, scaler);
}
```
关于 softmax 的计算笔者之前写过不少文章介绍过，是神经网络模型中的一个常用的算子。在 `blockSoftmaxKernel` 中考虑到有时 `seq_len_k` 会远大于 `block_size` 从而导致多次读取全局内存中的数据带来的延迟问题，这里笔者利用了共享内存缓存一行数据，以备下次读取。整体思路就是，先 block 内规约求最大值，让每个分量减去最大值防止后续指数运算溢出，再 block 内规约求和，最后对每个分量归一化。这里有兴趣的读者可以阅读笔者之前的两篇文章，进一步了解 softmax 的优化加速技巧。
- [【CUDA编程】OneFlow Softmax 算子源码解读之WarpSoftmax](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484512&idx=1&sn=b7d82ee936fd383a2ebb1dff4906a7c1&chksm=e92781d9de5008cf4a9892ee869fd46c6a9ff1bae270deec21675695838fa708cde6d65aa255&token=1862340673&lang=zh_CN#rd)
- [【CUDA编程】OneFlow Softmax算子源码解读之BlockSoftmax](https://mp.weixin.qq.com/s?__biz=MzIzOTY1NDEwMg==&mid=2247484529&idx=1&sn=1489f08958e00b9b03703c38fe645327&chksm=e92781c8de5008de5698acb1c8540cac373149e22011badb6de8a606e8631b681fb3ee23cd6e&token=1862340673&lang=zh_CN#rd)

### 3.9 QK*V 乘法
根据 Attention 计算公式，softmax 之后就开始计算 `QK*V`，这里 QK 的形状为 `[batch_size, head_num, seq_len_q, size_len_k]`，在 2、3 两个维度上进行乘法运算，可以看出不同 decoding 阶段的乘法形式有所不同，在 generation 阶段，`seq_len_q` 的值为 `1`，所以 `QK*V` 实际上是一个 GEMV 运算，而在 prompt 阶段，`QK*V` 是一个 GEMM 运算，同上面的 `Q*K` 运算一样，笔者进行了一个简单判断，分别调用不同的 cuBLAS API。

在 prompt 阶段，V 直接就使用前面 `DequantizedVTransposeKernel` 的计算结果 `value_out_fp_buf_` 进行矩阵运算即可，参数设置比较常规，具体参见源码。

在 generation 阶段，GEMV 的 M 就是 `value_cache`，V 则是 `qk_buf_`，调用 `cublasSgemvStridedBatched` API 传参计算即可。

### 3.10 AttnQuantizedTransposeKernel
多头注意力计算完成之后下一步就是拼接多头和投影变换，即 transpose 和 GEMM 操作，在 GEMM 之前还隐含了一个 INT8 量化操作，笔者将 transpose 和 INT8 量化融合在一个 Kernel 中，GEMM 操作仍然调用 cuBLAS API 完成。

```cpp
/** 量化、转置
* grid(seq_len, batch_size) block(32 * head_num)
* attn_buf:[batch_size, seq_len, head_num, size_per_head]
* attn:[batch_size, head_num, seq_len, size_per_head]
* attn_out_scale:[batch_size, seq_len]
*/
__global__ void warpAttnQuantizedTransposeKernel(int8_t *__restrict__ attn_buf, const float *__restrict__ attn,
                                                    float *__restrict__ attn_out_scale, const int batch_size,
                                                    const int head_num, const int seq_len, const int size_per_head)
{
    const int batch_id = blockIdx.y;
    const int seq_id = blockIdx.x;
    const int head_id = (threadIdx.x >> 5);
    const int tid = ((threadIdx.x & 0x1f) << 2);
    const int offset = batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head + seq_id * size_per_head;
    float4 val;
    char4 out_val;
    float absmax = -1e9f;
    int target_idx;
    float4 *inp_ptr = (float4 *)(attn + offset);
    char4 *out_ptr = (char4 *)attn_buf;
    if (tid < size_per_head)
    {
        val = inp_ptr[(threadIdx.x & 0x1f)];
        absmax = max(absmax, max(fabsf(val.x), max(fabsf(val.y), max(fabsf(val.z), fabsf(val.w)))));
        __syncthreads();
        absmax = blockAllReduceMax<float>(absmax);
        if (tid == 0)
        {
            attn_out_scale[batch_id * seq_len + seq_id] = absmax / 127.0f;
        }

        out_val.x = float_to_int8_rn(val.x * 127.0f / absmax);
        out_val.y = float_to_int8_rn(val.y * 127.0f / absmax);
        out_val.z = float_to_int8_rn(val.z * 127.0f / absmax);
        out_val.w = float_to_int8_rn(val.w * 127.0f / absmax);

        target_idx = batch_id * seq_len * head_num * size_per_head + seq_id * head_num * size_per_head + head_id * size_per_head + tid;
        out_ptr[target_idx >> 2] = out_val;
    }
}
```
以 `head_num <= 32 && size_per_head <= 128` 为例，`warpAttnQuantizedTransposeKernel` 核函数总共完成量化、转置两个操作。首先定义 block 为 `(32 * head_num)`，也就是说每个 warp（32 个线程）完成 `size_per_head` 个元素的计算任务，正好一个 block 内完成 `hidden_units` 个元素的计算任务。grid 定义为 `(seq_len,  batch_size)`。由于 `size_per_head` 不超过 `128` 所以 warp 内每个线程一次性读取 4 个元素（借助向量类型 `float4`）是足够一次读取完 `size_per_head` 个元素的，然后总共有 `head_num` 个 warp，每个 warp 负责一个 head，正好 block 内就把 `hidden_units` 个数据全部处理。简单来说，`0-31` 号线程处理 `head_id` 为 `0` 的 `size_per_head` 个元素，`32-63` 号线程处理 `head_id` 为 `1` 的 `size_per_head` 个元素，依次类推，这样一个 block 就能读完整行 `hidden_units` 个元素，可以直接 block 内规约求最大值用于量化计算过程。

以上设计思路的出发点就是：量化是在所有 head 的尺度上完成的，所以为了更方便地利用 block 内共享内存数据交换的机制，要在一个 block 内把所有数据全部读取出来。

当 `head_num` 或 `size_per_head` 比较大时，还有一种设计思路，即在一个线程内循环读取所有 head 在此分量上的数据，找出最大值，然后再 block 内规约即可，具体看下面的 Kernel 代码。

```cpp
/** 量化、转置
* grid(seq_len, batch_size) block(size_per_head)
* attn_buf:[batch_size, seq_len, head_num, size_per_head]
* attn:[batch_size, head_num, seq_len, size_per_head]
* attn_out_scale:[batch_size, seq_len]
*/
__global__ void blockAttnQuantizedTransposeKernel(int8_t *__restrict__ attn_buf, const float *__restrict__ attn,
                                                    float *__restrict__ attn_out_scale, const int batch_size,
                                                    const int head_num, const int seq_len, const int size_per_head)
{
    const int batch_id = blockIdx.y;
    const int seq_id = blockIdx.x;
    int offset;
    float val;
    int8_t out_val;
    float absmax = -1e9f;
    int target_idx;

    for (int head_id = 0; head_id < head_num; ++head_id)
    {
        offset = batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head + seq_id * size_per_head;
        val = attn[offset + threadIdx.x];
        absmax = max(absmax, fabsf(val));
    }
    __syncthreads();
    absmax = blockAllReduceMax<float>(absmax);

    if (threadIdx.x == 0)
    {
        attn_out_scale[batch_id * seq_len + seq_id] = absmax / 127.0f;
    }

    for (int head_id = 0; head_id < head_num; ++head_id)
    {
        offset = batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head + seq_id * size_per_head;
        val = attn[offset + threadIdx.x];
        out_val = float_to_int8_rn(val * 127.0f / absmax);
        target_idx = batch_id * seq_len * head_num * size_per_head + seq_id * head_num * size_per_head + head_id * size_per_head + threadIdx.x;
        attn_buf[target_idx] = out_val;
    }
}
```

### 3.11 Attention Output GEMM
Attention Output GEMM 操作是一个 INT8 * INT8 -> INT32 的矩阵乘法，直接调用 cuBLAS API 完成，复用 `from_tensor_int8_buf_` 和 `query_out_buf_` 存储输入和输出矩阵，具体参数设置参见源码。

### 3.12 DequantizedResidualResNormQuantized
Attention 计算完成后就会进入残差结构和 FFN 结构的部分，残差结构的结果将作为 FFN 结构的输入并且也会参与到 FFN 结构后续残差计算，所以要暂存起来。而在 FFN 结构中首先还要进行 ResNorm 操作和量化操作，量化完的结果也要暂存起来，所以这里会有两个需要输出的矩阵。为了最大程度减少 Kernel 启动开销，笔者将 Dequantized、Attention Residual、ResNorm、Quant 等操作全部融合到一个 Kernel 中实现。

```cpp
/**反量化、残差结构、ResNorm、量化
* grid(seq_len * batch_size) block(128)
* norm_out: [batch_size, seq_len, hidden_units]
* ffn_tensor: [batch_size, seq_len, hidden_units]
* from_temsor: [batch_size, seq_len, hidden_units]
* attn_out: [batch_size, seq_len, hidden_units]
* attn_out_scale: [batch_size, seq_len]
* attn_weight_scale: [hidden_units]
* gamma: [hidden_units]
*/
template <typename DataType>
__global__ void dequantizedResidualResNormQuantizedKernel(int8_t *__restrict__ norm_out, DataType *__restrict__ ffn_tensor, const DataType *__restrict__ from_temsor,
                                                            const int32_t *__restrict__ attn_out, const float *__restrict__ attn_out_scale, const float *__restrict__ attn_weight_scale,
                                                            const DataType *__restrict__ gamma, float *__restrict__ norm_scale, const float eps, const int hidden_units)
{
    const int row_id = blockIdx.x;
    const int offset = row_id * hidden_units;
    const float attn_scale_val = __ldg(attn_out_scale + row_id);

    extern __shared__ float s_buf[]; // hiddent_units
    float val;
    float mean = 0.0f;
    float absmax = -1e9f;
    char4 out_val;

    for (int tid = threadIdx.x; tid < hidden_units; tid += blockDim.x)
    {
        val = static_cast<float>(attn_out[offset + tid]) * attn_scale_val * __ldg(attn_weight_scale + tid) + static_cast<float>(from_temsor[offset + tid]);
        s_buf[tid] = val;
        ffn_tensor[offset + tid] = static_cast<DataType>(val);
        mean += val * val;
        absmax = max(absmax, fabsf(val * static_cast<float>(__ldg(gamma + tid))));
    }
    __syncthreads();

    mean = blockAllReduceSum<float>(mean / hidden_units);
    mean = rsqrtf(mean + eps);

    absmax = blockAllReduceMax<float>(absmax);
    absmax *= mean;
    if (threadIdx.x == 0)
    {
        norm_scale[blockIdx.x] = absmax / 127.0f;
    }

    int target_idx;
    char4 *out_ptr = (char4 *)norm_out;
    for (int tid = (threadIdx.x << 2); tid < hidden_units; tid += (blockDim.x << 2))
    {
        out_val.x = float_to_int8_rn(s_buf[tid] * mean * static_cast<float>(__ldg(gamma + tid)) * 127.0f / absmax);
        out_val.y = float_to_int8_rn(s_buf[tid + 1] * mean * static_cast<float>(__ldg(gamma + tid + 1)) * 127.0f / absmax);
        out_val.z = float_to_int8_rn(s_buf[tid + 2] * mean * static_cast<float>(__ldg(gamma + tid + 2)) * 127.0f / absmax);
        out_val.w = float_to_int8_rn(s_buf[tid + 3] * mean * static_cast<float>(__ldg(gamma + tid + 3)) * 127.0f / absmax);
        target_idx = row_id * hidden_units + tid;
        out_ptr[target_idx >> 2] = out_val;
    }
}

template <typename DataType>
void launchDequantizedResidualResNormQuantized(int8_t *norm_out, DataType *__restrict__ ffn_tensor, const DataType *from_temsor, const int32_t *attn_out, const float *attn_out_scale,
                                                const float *attn_weight_scale, const DataType *gamma, float *norm_scale, const float eps, const int rows, const int hidden_units, cudaStream_t stream)
{
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    assert(hidden_units % 4 == 0);
    int mem_size = hidden_units * sizeof(float);
    dequantizedResidualResNormQuantizedKernel<DataType><<<rows, 128, mem_size, stream>>>(norm_out, ffn_tensor, from_temsor, attn_out, attn_out_scale, attn_weight_scale, gamma,
                                                                                            norm_scale, eps, hidden_units);
}
```
可以看出，这个 `dequantizedResidualResNormQuantizedKernel` 与前面介绍的 `ResNormQuantizedKernel` 相比只是多了反量化与残差操作，其中反量化和残差逻辑体现在这一行代码：
```cpp
val = static_cast<float>(attn_out[offset + tid]) * attn_scale_val * __ldg(attn_weight_scale + tid) + static_cast<float>(from_temsor[offset + tid]);
```
反量化和残差结束后，计算得到的 `val` 就是 FFN 结构的输入，为了后续 FFN 残差计算，先暂存到 `ffn_tensor` 中，其他的代码逻辑与 `ResNormQuantizedKernel` 基本一致，这里不再赘述。

```cpp
ffn_tensor[offset + tid] = static_cast<DataType>(val);
```

### 3.13 W1 GEMM 和 W3 GEMM
从 Opendecoder 架构示意图可以看出，ResNorm 后紧接着就是 W1 GEMM 和 W3 GEMM，隐藏层单元数量为 `ffn_hidden_units` 通常要大于 `hidden_units`，这两个 GEMM 就是普通的 INT8 * INT8 -> INT32 GEMM，直接调用 cuBLAS API 计算即可，参数设置参见源码。

### 3.14 DequantizedSiluMultifyQuantized
W1 GEMM 和 W3 GEMM 完成之后，对两个输出矩阵进行反量化，然后对 `w1_out` 进行 Silu 激活，接着对 `w3_out` 和激活后的 `w1_out` 进行 PointWise 乘法，对结果量化后作为 W2 GEMM 的输入矩阵。下面先来看一下 Silu 函数的公式：
$$
Silu(x) = \frac{x}{1 + e^{-x}} = x \cdot Sigmod(x)
$$

```cpp
/** 反量化、silu、element-wise-multify、量化
* grid(nrows) block(128)
* out_buf: [nrows, hidden_units]
* w1_ret w3_ret: [nrows, hidden_units]
* norm_scale: [nrows, ]
* w1_weight_scale w3_weight_scale: [hidden_units, ]
* out_scale: [nrows, ]
*/
__global__ void dequantizedSiluMultifyQuantizedKernel(int8_t *__restrict__ out_buf, const int32_t *__restrict__ w1_ret, const float *__restrict__ norm_scale,
                                                        const float *__restrict__ w1_weight_scale, const int32_t *__restrict__ w3_ret, const float *__restrict__ w3_weight_scale,
                                                        float *__restrict__ out_scale, const int hidden_units)
{
    const int row_id = blockIdx.x;
    const float norm_scale_val = __ldg(norm_scale + row_id);
    const int offset = row_id * hidden_units;
    extern __shared__ float s_buf[];
    float val;
    float absmax = -1e9f;
    for (int tid = threadIdx.x; tid < hidden_units; tid += blockDim.x)
    {
        val = static_cast<float>(w1_ret[offset + tid]) * norm_scale_val * __ldg(w1_weight_scale + tid);
        val = silu(val);
        val *= static_cast<float>(w3_ret[offset + tid]) * norm_scale_val * __ldg(w3_weight_scale + tid);
        s_buf[tid] = val;
        absmax = max(absmax, fabsf(val));
    }
    __syncthreads();

    absmax = blockAllReduceMax<float>(absmax);
    if (threadIdx.x == 0)
    {
        out_scale[row_id] = absmax / 127.0f;
    }

    float scale_val = 127.0f / absmax;
    char4 out_val;
    char4 *out_ptr = (char4 *)out_buf;
    for (int tid = (threadIdx.x << 2); tid < hidden_units; tid += (blockDim.x << 2))
    {
        out_val.x = float_to_int8_rn(s_buf[tid] * scale_val);
        out_val.y = float_to_int8_rn(s_buf[tid + 1] * scale_val);
        out_val.z = float_to_int8_rn(s_buf[tid + 2] * scale_val);
        out_val.w = float_to_int8_rn(s_buf[tid + 3] * scale_val);
        out_ptr[(offset + tid >> 2)] = out_val;
    }
}

void launchDequantizedSiluMultifyQuantized(int8_t *out_buf, const int32_t *w1_ret, const float *norm_scale, const float *w1_weight_scale,
                                            const int32_t *w3_ret, const float *w3_weight_scale, float *out_scale, const int nrows, const int hidden_units, cudaStream_t stream)
{
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    assert(hidden_units % 4 == 0);
    int mem_size = sizeof(float) * hidden_units;
    dequantizedSiluMultifyQuantizedKernel<<<nrows, 128, mem_size, stream>>>(out_buf, w1_ret, norm_scale, w1_weight_scale,
                                                                            w3_ret, w3_weight_scale, out_scale, hidden_units);
}
```
其中，反量化、Sliu、PointWise Multify 的计算逻辑都在这三行代码中完成：

```cpp
val = static_cast<float>(w1_ret[offset + tid]) * norm_scale_val * __ldg(w1_weight_scale + tid);
val = silu(val);
val *= static_cast<float>(w3_ret[offset + tid]) * norm_scale_val * __ldg(w3_weight_scale + tid);
```

把上述计算结果存入共享内存变量 `s_buf` 中以备量化时读取，量化的计算逻辑前面已经介绍过，这里不再重复介绍了。这个 Kernel 中分别利用了共享内存、块内规约、向量化数据类型等优化加速技巧，比较常规。

### 3.15 W2 GEMM
W2 GEMM 隐藏层单元数量为 `hidden_units`，相当于对输入 tensor 进行了一次压缩，又回到了原来的形状，这个 GEMM 就是普通的 INT8 * INT8 -> INT32 GEMM，直接调用 cuBLAS API 计算即可，参数设置参见源码。

### 3.16 DequantizedResidual
W2 GEMM 之后就是残差结构，由于 W2 GEMM 是一个 INT8 乘法，因此还隐含了一个反量化操作，笔者把这两个操作融合到了一个 Kernel，计算逻辑比较简单，不再详细介绍，至此，一个 Decoder Layer 的计算过程已经完成。
```cpp
/**反量化、残差结构
* grid(seq_len * batch_size) block(128)
* out: [batch_size, seq_len, hidden_units]
* ffn_tensor: [batch_size, seq_len, hidden_units]
* from_temsor: [batch_size, seq_len, hidden_units]
* inp: [batch_size, seq_len, hidden_units]
* inp_scale: [batch_size, seq_len]
* weight_scale: [hidden_units]
*/
template <typename DataType>
__global__ void dequantizedResidualKernel(DataType *__restrict__ out, const DataType *__restrict__ from_temsor,
                                            const int32_t *__restrict__ inp, const float *__restrict__ inp_scale, const float *__restrict__ weight_scale,
                                            const int hidden_units)
{
    const int row_id = blockIdx.x;
    const int offset = row_id * hidden_units;
    const float inp_scale_val = __ldg(inp_scale + row_id);
    float val;

    for (int tid = threadIdx.x; tid < hidden_units; tid += blockDim.x)
    {
        val = static_cast<float>(inp[offset + tid]) * inp_scale_val * __ldg(weight_scale + tid) + static_cast<float>(from_temsor[offset + tid]);
        out[offset + tid] = static_cast<DataType>(val);
    }
}
```

## 4 Decoding 模型
前面介绍过，针对 Decoding 模型的解码场景，FasterLLaMA v1.0 提供了两种基于采样解码的解码方案：top-k 解码和 top-p 解码，具体逻辑见源码的 decoding_sampling.h、decoding_sampling.cu 两个文件。
### 4.1 DecodingSampling 类结构


```cpp
template <OperationType OpType_>
class DecodingSampling
{
private:
    typedef DecoderTransformerTraits<OpType_> Traits_;
    typedef typename Traits_::DataType DataType_;
    const IAllocator &allocator_;
    struct DecodingSamplingArguments args_;

    const cublasComputeType_t computeType_ = Traits_::computeType;
    const cudaDataType_t AType_ = Traits_::AType;
    const cudaDataType_t BType_ = Traits_::BType;
    const cudaDataType_t CType_ = Traits_::CType;
    int cublasAlgo_[1] = {20};

    OpenDecoder<OpType_, OperationType::INT8> *decoder_;
    float *K_cache_;
    float *V_cache_;
    DataType_ *from_tensor_[2];
    char *decoder_buf_;
    DataType_ *decoder_normed_result_buf_;
    float *logits_buf_;
    float *step_logits_buf_;
    int *word_ids_buf_;
    bool *finished_buf_;
    int *topk_ids_buf_;
    float *topk_val_buf_;
    void *buf_;
    bool *h_finished_buf_;
    // is initialized by [[0, 1, ..., vocab_size-1], [0, 1, ..., vocab_size-1], ..., [0, 1, ..., vocab_size-1]]
    int *topp_id_vals_buf_;
    float *topp_sorted_logits_prob_buf_;
    int *topp_sorted_id_vals_buf_;
    // is initialized by [0, vocab_size, ..., batch_size * vocab_size]
    int *topp_offset_buf_;

    void *temp_storage_;

public:
    DecodingSampling(const IAllocator &allocator, const int batch_size,
                        const int max_prompt_len, const int max_gen_len,
                        const int head_num, const int size_per_head,
                        const int vocab_size, const int decoder_layers,
                        const int end_id, const int ffn_hidden_units,
                        const int candidate_num = 0, const float probability_threshold = 0.0);

    void forward(const DecoderInitParam<DataType_, int8_t> *param,
                    DecodingInitParam<DataType_> decoding_params);
    

    virtual ~DecodingSampling();
};
```
`DecodingSampling` 类模版有 1 个模版参数：`OpType_` 指明了激活值的数据类型，通常就是 FP32、FP16。

除了一些模型结构、矩阵乘法参数相关的成员变量以外，主要就是声明了几个变量 buffer，具体用处和含义如下：
- `from_tensor_`：存储 `DataType` 类型的激活值矩阵，从声明可以看出这是个双缓冲区，会在不同的 Decoder Layer 中不断复用以节省设备内存，比如对于第一层 Decoder，`from_tensor_[0]` 是输入，`from_tensor_[1]` 是输出，对于第二层 Decoder，`from_tensor_[1]`（即第一层 Decoder 的输出）是输入，`from_tensor_[0]` 是输出，依次类推。
- `K_cache_` 和 `V_cache_`：存储 KV cache。
- `decoder_buf_`：就是前面 `Opendecoder` 的 `initialize` 函数的参数 `buf_`，即 `Opendecoder` 的缓冲区指针。
- `decoder_normed_result_buf_`：存储 Decoder 结束后的 ResNorm 的计算结果，用于 Output GEMM 的输入矩阵。
- `logits_buf_`：存储 logits。
- `step_logits_buf`：存储当前 step 的 logits。
- `word_ids_buf_`：存储所有 step 解码得到 word_id，注意这里的内存顺序为 `[steps, batch_size]`。
- `finished_buf_`：存储当前 step 的样本是否解码完成的标识。
- `topk_ids_buf_`：用于 top-k 采样的排序环节，存储 top-k 个 word_id 的排序。
- `topk_val_buf_`：用于 top-k 采样的排序环节，存储 top-k 个 word_id 对应的 logits 的排序。
- `buf_`：总缓冲区的指针。
- `topp_id_vals_buf_`：用于 top-p 采样的排序环节，存储所有 word_id 的序号，作为 CUB 排序 API 的参数，被初始化为 `[[0, 1, ..., vocab_size-1], [0, 1, ..., vocab_size-1], ..., [0, 1, ..., vocab_size-1]]`。
- `topp_sorted_logits_prob_buf_`：排序后的 `step_logits_buf`。
- `topp_sorted_id_vals_buf_`：根据 `step_logits_buf` 排序后的 `topp_id_vals_buf_`。
- `topp_offset_buf_`：用于 top-p 采样的排序环节，作为 CUB 排序 API 的参数，被初始化为 `[0, vocab_size, ..., batch_size * vocab_size]`。
- `temp_storage_`：用于 top-p 采样的排序环节，作为 CUB 排序 API 的参数，临时缓冲区。

除了一些成员变量以外，`DecodingSampling` 类模版的核心逻辑就在成员方法 `forward` 中，而 `DecodingSampling` 构造方法中主要是进行一些成员变量初始化、设备内存分配、指针偏移量计算等准备工作，具体见源码。

### 4.2 初始化 Kernel
在 Decoding 之前需要对一些后续解码环节需要用到的变量进行初始化，比如要对 `finished_buf_` 向量中的元素全部初始化为 `false`，对每个样本的生成长度 `decoding_params.sequence_length` 初始化为 `0`，此外针对 top-p 解码的排序环节还需要对一些辅助参数变量进行初始化，如 `topp_id_vals_buf_`、`topp_offset_buf_` 等，所以笔者这里提供了两个初始化 Kernel，根据不同的解码方法选择不同的 Kernel 执行。
```cpp
if (args_.candidate_num_ != 0)
{
    /**
    * decoding_params.sequence_length is initialized by 0
    * finished_buf_ is initialized by false
    */
    launchTopKSamplingInitKernel(finished_buf_, decoding_params.sequence_length, args_.batch_size_, decoding_params.stream);
}
else if (args_.probability_threshold_ != 0.0)
{
    /**
    * decoding_params.sequence_length is initialized by 0
    * finished_buf_ is initialized by false
    * topp_offset_buf is initialized by [0, vocab_size, ..., batch_size * vocab_size]
    * topp_id_val_buf is initialized by [[0, 1, ..., vocab_size-1], [0, 1, ..., vocab_size-1], ..., [0, 1, ..., vocab_size-1]]
    */
    launchTopPInitializationKernel(finished_buf_, decoding_params.sequence_length, topp_id_vals_buf_, topp_offset_buf_,
                                    args_.batch_size_, args_.vocab_size_, decoding_params.stream);
}
```
两个初始化 Kernel 的实现逻辑比较简单，这里就不再介绍了，另外，也可以把这些变量放进 `decoding_params` 中由外部传入。

### 4.3 计算 freqsCis 矩阵
freqsCis 矩阵是用于 Rope 计算的参数矩阵，当模型超参数 `size_per_head` 和 `total_len` 确定后，矩阵元素值就固定了，所以这里需要计算一下 `total_len` 的值。首先 `total_len` 肯定是可以直接取 `max_prompt_len_ + max_gen_len_` 的，也就是直接初始化一个最大的 freqsCis 矩阵，但是这样带来一个问题，很多时候我们的 prompt_tokens 是达不到 `max_prompt_len_` 的，直接初始化一个最大的 freqsCis 矩阵会浪费不少设备内存。为了尽量节省设备内存，可以计算一下当前 prompt_tokens 中的最大长度 `max_prompt_seq_len`，然后用 `max_prompt_seq_len + max_gen_len_` 作为最大长度。

关于 freqsCis 矩阵的计算逻辑已经在 3.4 中进行了介绍，这里不再重复。

### 4.4 EmbeddingLookupKernel
词嵌入的过程就是从 prompt_tokens 中取出 word_id 取 `embedding_table` 中查表映射的过程，一个 word_id 映射为一个长度为 `hidden_units` 的向量。根据不同的 Decoding 阶段，其输入矩阵参数有所不同，在 prompt 阶段，输入是 `decoding_params.prompt_tokens`，而在 generation 阶段输入则是暂存在 `word_ids_buf_` 中的上一个 step 解码生成的 word_id，其形状为 `[batch_size, 1]`。笔者提供了一个函数 `launchEmbeddingLookupKernel` 可以兼容两种情况，只是输出参数需要根据情况调整。

```cpp
 template <typename T>
__global__ void embeddingLookupKernel(T *__restrict__ from_tensor, const T *__restrict__ embedding_table,
                                        const int *__restrict__ word_ids, const int max_len, const int hidden_units)
{
    const int token_id = blockIdx.x;
    const int batch_id = blockIdx.y;
    int write_pos, lookup_pos;
    for (int tid = threadIdx.x; tid < hidden_units; tid += blockDim.x)
    {
        write_pos = tid + token_id * hidden_units + batch_id * gridDim.x * hidden_units;
        lookup_pos = word_ids[batch_id * max_len + token_id] * hidden_units + tid;
        // 1. lookup the table
        // 2. multiply hidden_dim**0.5
        from_tensor[write_pos] = embedding_table[lookup_pos] * (T)sqrtf(float(hidden_units));
    }
}

template <>
__global__ void embeddingLookupKernel(half *__restrict__ from_tensor, const half *__restrict__ embedding_table,
                                        const int *__restrict__ word_ids, const int max_len, const int hidden_units)
{
    const int token_id = blockIdx.x;
    const int batch_id = blockIdx.y;
    int write_pos, lookup_pos;
    for (int tid = threadIdx.x; tid < hidden_units; tid += blockDim.x)
    {
        write_pos = tid + token_id * hidden_units + batch_id * gridDim.x * hidden_units;
        lookup_pos = word_ids[batch_id * max_len + token_id] * hidden_units + tid;
        // 1. lookup the table
        // 2. multiply hidden_dim**0.5
        from_tensor[write_pos] = __float2half(__half2float(embedding_table[lookup_pos]) * sqrtf(float(hidden_units)));
    }
}

template <typename T>
void launchEmbeddingLookupKernel(T *__restrict__ from_tensor, const T *__restrict__ embedding_table, const int *__restrict__ word_ids,
                                    const int batch_size, const int cur_seq_len, const int max_len, const int hidden_units,
                                    cudaStream_t stream)
{
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    dim3 grid(cur_seq_len, batch_size);
    dim3 block(256);
    embeddingLookupKernel<T><<<grid, block, 0, stream>>>(from_tensor, embedding_table, word_ids, max_len, hidden_units);
}

...

/**
* Embedding Lookup
*/
if (cur_pos == min_prompt_seq_len)
{
    #ifndef NDEBUG
    printf("[FL][INFO] prompt tokens embedding lookup\n");
    #endif
    // prompt phase, prompt_tokens[:, :cur_pos] is embedded to from_tensor which shape is [batch_size, cur_seq_len, hidden_units]
    launchEmbeddingLookupKernel(from_tensor_[0], decoding_params.embedding_table, decoding_params.prompt_tokens,
                                            args_.batch_size_, cur_seq_len, decoding_params.max_prompt_seq_len, 
                                            args_.hidden_units_, decoding_params.stream);
}
else
{
#ifndef NDEBUG
    printf("[FL][INFO] step: %d tokens embedding lookup\n", step);
#endif
    // generation phase, word_ids_buf_ is embedded to from_tensor which shape is [batch_size, hidden_units]
    launchEmbeddingLookupKernel(from_tensor_[0], decoding_params.embedding_table,
                                            word_ids_buf_ + (step - 2) * args_.batch_size_,
                                            args_.batch_size_, 1, 1, args_.hidden_units_, decoding_params.stream);
}
```
从代码中可以看出，主要就是两步：查表、缩放。在查表阶段，需要根据 token 在原矩阵中的位置计算偏移量，这块要特别注意源矩阵的宽度。在缩放阶段，会把查表得到的向量中的每个分量乘以 $\sqrt{d}$，$d$ 就是 `hidden_units`，这个缩放步骤可以根据需要去除。

### 4.5 ResNorm
在 EmbeddingLookup 结束之后就是进行多层 Decoder Layers 运算，Decoder 结束后紧接着进行一次 ResNorm 操作，关于 ResNorm 的计算过程在 3.2 中的 `ResNormQuantizedKernel` 也有过详细介绍，这里不需要量化，计算过程更加简洁，代码如下：
```cpp
/** resNorm
* grid(batch_size * seq_len)  block(128)
* output: [batch_size, seq_len, hidden_units]
* input: [batch_size, seq_len, hidden_units]
* gamma: [hidden_units, ]
*/
template <typename DataType>
__global__ void resNormKernel(DataType *__restrict__ output, const DataType *__restrict__ input,
                                const DataType *__restrict__ gamma, const float eps, const int hidden_units)
{
    const int offset = blockIdx.x * hidden_units;
    float mean;
    float val = 0.0f;
    for (int i = threadIdx.x; i < hidden_units; i += blockDim.x)
    {
        val += input[offset + i] * input[offset + i];
    }
    __syncthreads();

    val = blockAllReduceSum<float>(val);
    mean = rsqrtf(val / hidden_units + eps);
    // __syncthreads();

    for (int i = threadIdx.x; i < hidden_units; i += blockDim.x)
    {
        output[offset + i] = (DataType)(mean * input[offset + i] * gamma[i]);
    }
}
```
每个 block 处理一行元素，第一层循环配合块内规约求 RMS，第二层给每个分量归一化，比较常规。

### 4.6 Output GEMM
这一层是神经网络模型的输出层，即将形状如 `[batch_size, seq_len, hidden_units]` 的 tensor 线性变换为形状如 `[batch_size, seq_len, vocab_size]` 的 logits 矩阵，这里笔者直接调用了 cuBLAS API 完成矩阵乘法，这个 GEMM 不涉及 INT8 量化，具体参数设置见源码。

### 4.7 top-k 采样
获得 logits 之后就需要进行解码，解码的目的是生成当前 step 对应的 token。对于 prompt 阶段，生成的 logits 形状为 `[batch_size, min_prompt_seq_len, vocab_size]`，其实只有最后一个 token 对应的结果用于解码，所以需要单独取出来存入 `step_logits_buf_` 中，即 `step_logits_buf_ = logits_buf[:, -1, :]`。此外还要做一次停止符判断，即如果当前 step 下当前样本已经解码结束，则把 `step_logits_buf_` 中 `end_id` 对应的分量设置为一个极大的值，使得后续 softmax 环节这个分量足够大，最终采样采到 `end_id`。

```cpp
/** 取 logits[:, -1, :] 存入 step_logits，并顺便进行停止符判断
* grid(batch_size), block(min(vocab_size, 1024))
* step_logits: [batch_size, 1, vocab_size]
* logits: [batch_size, seq_len, vocab_size]
* finished: [batch_size, 1]
*/
__global__ void updateLogitsWithoutSoftmax(float *__restrict__ step_logits, const float *__restrict__ logits, const int end_id,
                                            const bool *__restrict__ finished, const int seq_len, const int vocab_size)
{
    const bool is_finished = finished[blockIdx.x];

    for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x)
    {
        int idx = blockIdx.x * seq_len * vocab_size + (seq_len - 1) * vocab_size + tid;
        if (is_finished)
        {
            step_logits[blockIdx.x * vocab_size + tid] = (tid == end_id) ? FLT_MAX : -1 * FLT_MAX;
        }
        else
        {
            step_logits[blockIdx.x * vocab_size + tid] = logits[idx];
        }
    }
}

void launchUpdateLogitsWithoutSoftmax(float *__restrict__ step_logits, const float *__restrict__ logits, const int end_id,
                                        const bool *__restrict__ finished, const int batch_size, const int seq_len,
                                        const int vocab_size, cudaStream_t stream)
{
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    dim3 grid(batch_size);
    dim3 block(min(vocab_size, 1024));
    /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
    updateLogitsWithoutSoftmax<<<grid, block, 0, stream>>>(step_logits, logits, end_id, finished, seq_len, vocab_size);
}
```

获取到 `step_logits` 之后就需要进行 top-k 采样解码了，具体来说，在每个 step，首先选择概率最高的 k 个 word 作为候选 word 构成一个集合，然后将这个子集中 word 的概率再归一化，最后从新的概率分布中采样。所以第一步就是要在形状为 `[batch_size, vocab_size]` 的`step_logits` 矩阵中分 `batch_size` 组选出 k 个最大的 logits 对应的 word。对于每个样本来说就是 `vocab_size` 个元素中找 topk 个元素，目前 k 值仅支持 1、2、4 三个取值。

TopK 问题是一个经典算法问题，通常我们通过维护一个小根堆，堆里存了 k 个数据，每次新数据跟堆顶数据比较，大于堆顶元素就替换掉堆顶元素，然后重新建堆，遍历完所有元素后，堆中元素就是 TopK。这里也使用了这个思路，但是并没有使用堆结构，而是定义了一个结构体 `TopK`，我们来看一下这个结构体。
```cpp
template<typename T, int MAX_K>
struct TopK
{
    int p[MAX_K];
    T u[MAX_K];

    __device__ __forceinline__ void insert(T elem, int elem_id)
    {
        // 把插入元素跟最后一个元素比较，如果插入元素更大，则替换掉最后一个元素
        if (elem > u[MAX_K-1] || (p[MAX_K-1] == -1) || ((elem == u[MAX_K-1]) && (elem_id < p[MAX_K-1])))
        //if (elem > u[MAX_K-1] || ((elem == u[MAX_K-1]) && (elem_id < p[MAX_K-1])))
        {
            u[MAX_K-1] = elem;
            p[MAX_K-1] = elem_id;
        }
        // 冒泡排序，把 TopK 中的元素进行排序
        for(int k = MAX_K - 2; k >= 0; --k)
        {
            if ((u[k+1] > u[k]) || (p[k] == -1) || ((u[k+1] == u[k])&&(p[k+1] < p[k])))
            //if ((u[k+1] > u[k]) || ((u[k+1] == u[k])&&(p[k+1] < p[k])))
            {
                T u2 = u[k];
                int p2 = p[k]; 
                u[k] = u[k+1];
                p[k] = p[k+1];
                u[k+1] = u2;
                p[k+1] = p2;
            }
        }
    }

    __device__ __forceinline__ void init()
    {
      #pragma unroll
      for(int i = 0; i < MAX_K; i++)
      {
        p[i] = -1;
        u[i] = -FLT_MAX;
      }
    }
};
```
可以看到，结构体中有两个长度为 `MAX_K` 的数组变量，`p` 用来存索引，`u` 用来存值，一一对应并按值降序排列。为啥弄两个数组？是因为这里我们还需要元素的位置，也就是 `word_id`，这两个数组同步更新。除了成员变量以外还有两个成员函数，一个是初始化函数 `init` 主要用来初始化 `p` 和 `u`，另一个是 `insert` 函数用来“插入元素”和“建堆”。`insert` 函数中首先比较最后一个元素和新插入元素，满足以下任意条件后，将用新插入的元素替换掉 `TopK` 中最后一个元素。
- 插入元素大于最后一个元素
- 最后一个元素是初始化的标识，也就是数组没有满
- 插入元素等于最后一个元素，但是插入元素的索引更小

插入元素后，还得“建堆”保证堆顶元素最小，这里直接用排序代替“建堆”，k 值比较小直接冒泡排序即可，排序完成后，数组中的元素恢复降序排列。  

`TopK` 结构介绍完之后，下面就是如何使用 `TopK` 结构完成对 `step_logits` 的求 TopK 操作。使用 `beam_topK_kernel` 核函数来求 TopK，`grid_size` 设置为 `batch_size`，`block_size` 设置为 `256`，也就是说一个 block 内要处理 `vocab_size` 个元素，从中选出 TopK，每个线程处理 `vocab_size / 256` 个元素，步长为 `256`。

```cpp
template <typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__
void beam_topK_kernel(const T *__restrict__ log_probs,
                        int *__restrict__ topk_tmp_id_buf,
                        T *__restrict__ topk_tmp_val_buf,
                        const int vocab_size,
                        T diversity_rate)
{
    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    TopK<T, MAX_K> partial;

#pragma unroll
    for (int i = 0; i < MAX_K; ++i)
    {
        partial.p[i] = -1;
        partial.u[i] = -FLT_MAX;
    }

#pragma unroll
    for (int elem_id = thread_id; elem_id < vocab_size; elem_id += THREADBLOCK_SIZE)
    {
        int index = elem_id + block_id * vocab_size;
        partial.insert(log_probs[index], index);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if (thread_id == 0)
    {
        int index = block_id * MAX_K;

#pragma unroll
        for (int i = 0; i < MAX_K; ++i)
        {
            topk_tmp_id_buf[index + i] = total.p[i];
            topk_tmp_val_buf[index + i] = total.u[i] + diversity_rate * (T)i;
        }
    }
}
```
Kernel 内部首先使用 CUB 库进行了块内规约前的准备，这个我们暂且不去看，之后内部定义了一个寄存器变量 `partial`，`partial` 存储了当前线程处理元素的 TopK，相当于当前线程下的小根堆，随后对 `partial` 进行初始化，这块其实也可以直接调用成员函数 `init`。然后就是对当前线程待处理的元素进行遍历，让 `partial` 来 `insert` 待处理元素，全部 `insert` 一遍后的 `partial` 其实就存储了当前线程处理的所有元素的 TopK。但是我们的目标是要获取整个 block 内的全局 TopK，所以我们还需要进行一次“大合并”，把所有的 TopK 合并成一个，这实际相当于一次块内规约操作，只是我们还需要定义一个操作函数，显然这个操作函数的输入是两个 `TopK` 类型的变量，输出是 `TopK` 类型，其计算逻辑就是把两个 `TopK` 合并成 1 个 `TopK`。这里也提供了一个 `reduce_topk_op` 函数来完成这个任务。  
```cpp
template<typename T, int MAX_K>
__device__ __forceinline__ TopK<T, MAX_K> reduce_topk_op(const TopK<T, MAX_K>& a, const TopK<T, MAX_K>& b)
{
    TopK<T, MAX_K> res = a;
    for(int i = 0; i < MAX_K; ++i)
        res.insert(b.u[i], b.p[i]);
    return res;
}
```
可以看到，`reduce_topk_op` 是通过遍历一个 `TopK` 变量 `b` 的元素，不断 `insert` 到另一个 `TopK` 变量 `a` 的拷贝 `res` 中实现的合并工作。 

有了操作函数以后，直接调用 CUB 库的块内规约 API 就完成了块内规约，获取了整个 block 内的全局 TopK `total`。当 `thread_id == 0` 时，把这 `k` 个元素对应的 `logit` 和 `word_id` 写入 `topk_tmp_val_buf` 和 `topk_tmp_id_buf` 中。这里还有个 `diversity_rate` 参数，这是一个修正系数，实际设置为 `0.0f` 并没有启用。  

获取 TopK 之后，计算每个 word 的概率，然后在 TopK 中归一化，最后根据归一化后的概率采样。其实就是先 Softmax 后采样，我们来看一下源码。

```cpp
/**
* top-k Sampling kernel
* grid(1), block(batch_size)
*/
template <typename T>
__global__ void topKSampling(int *__restrict__ topk_tmp_id_buf, T *__restrict__ topk_tmp_val_buf, int *__restrict__ ids,
                                int *__restrict__ sequence_length, bool *__restrict__ finished_buf,
                                const int *__restrict__ prompt_tokens, const bool *__restrict__ prompt_tokens_mask,
                                const int cur_pos, const int max_prompt_seq_len, const int candidate_num,
                                const int random_num, const int end_id, const int batch_size, const int vocab_size)
{
    if (threadIdx.x < batch_size)
    {
        // prompt phase, next_token[:] = prompt_tokens[:, cur_pos]
        if (cur_pos < max_prompt_seq_len && prompt_tokens_mask[threadIdx.x * max_prompt_seq_len + cur_pos])
        {
            ids[threadIdx.x] = prompt_tokens[threadIdx.x * max_prompt_seq_len + cur_pos];
        }
        else
        {
            // The maximum number of k logits in the current batch
            float max_val = (float)topk_tmp_val_buf[threadIdx.x * candidate_num];

            float sum = 0.0f;
            float tmp_val;
            for (int i = 0; i < candidate_num; ++i)
            {
                tmp_val = __expf(topk_tmp_val_buf[threadIdx.x * candidate_num + i] - max_val);
                topk_tmp_val_buf[threadIdx.x * candidate_num + i] = tmp_val;
                sum += tmp_val;
            }

            curandState_t local_state;
            curand_init(random_num, threadIdx.x, 0, &local_state);
            float rand_num = curand_uniform(&local_state) * sum;

            ids[threadIdx.x] = topk_tmp_id_buf[threadIdx.x * candidate_num + candidate_num - 1] % vocab_size;
            for (int i = 0; i < candidate_num; i++)
            {
                rand_num = rand_num - topk_tmp_val_buf[threadIdx.x * candidate_num + i];
                if (rand_num <= 0.0f)
                {
                    ids[threadIdx.x] = topk_tmp_id_buf[threadIdx.x * candidate_num + i] % vocab_size;
                    break;
                }
            }

            sequence_length[threadIdx.x] = finished_buf[threadIdx.x] ? sequence_length[threadIdx.x] : sequence_length[threadIdx.x] + 1;
            finished_buf[threadIdx.x] = ids[threadIdx.x] == end_id ? true : false;
        }
    }
}

...

assert(batch_size <= 1024);
if (batch_size <= 128)
{
    local_block_size = 128;
}
else if (batch_size <= 256)
{
    local_block_size = 256;
}
else if (batch_size <= 512)
{
    local_block_size = 512;
}
else
{
    local_block_size = 1024;
}
topKSampling<T><<<1, local_block_size, 0, stream>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids, sequence_length, finished_buf,
                                                    prompt_tokens, prompt_tokens_mask, cur_pos, max_prompt_seq_len, candidate_num,
                                                    random_num, end_id, batch_size, vocab_size);
```
核函数中 `grid_size` 和 `block_size` 分别设置为 `1` 和 `batch_size`，一个线程完成一个样本的 top-k 采样。首先判断当前 Decoding 阶段，如果当前样本还处于 prompt 阶段，也就是说下一个 token 是已经存在的，那直接替换即可，即 `next_token[:] = prompt_tokens[:, cur_pos]`，否则的话就需要采样解码。

在解码的同时还要先计算 softmax，采样是要根据 softmax 值归一化的不是直接根据 logits 值。先根据索引从 `topk_tmp_val_buf` 中获取 TopK 中的最大值，然后让当前元素减去最大值然后求指数，再存入 `topk_tmp_val_buf`。在 0 号线程内循环求规约和，得到 `sum`，这时候其实已经可以开始采样了，没有必要非得归一化。调用 cuRand API 从均匀分布中随机一个 `0~1` 之间的数再乘以 `sum`，得到一个 `0~sum` 之间的数 `rand_num`，要知道 TopK 中各元素是降序排列的，我可以把他当成 k 个相互连接的组合线段记作 $S_t$（其中每个子线段记作 $S_i$），把 `rand_num` 当成一根长度为 `rand_num` 的线段记作 $S_r$，并将其与 $S_t$ 的最左侧对齐，那么 $S_r$ 的右端点落在 $S_t$ 的哪个子线段中就认为采样选中了哪个 word，笔者给出如下示意图。  
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5qKvGhrwW5bOI1eM5TrQ7XNZgacjbkJfrzDZC6ajPavay5PEZVmupvQWcJ2DW7ic3IcMtD3ZEd0n6A/640?wx_fmt=png)

随后根据采样选中的 `word_id` 对 `sequence_length` 和 `finished_buf_` 进行更新，至此当前 step 的采样解码就完成了。 

### 4.8 top-p 采样
相比于 top-k 方法从概率最高的 k 个候选词中采样，top-p 采样不再取一个固定的 k，而是固定候选集合的概率密度和在整个概率分布中的比例。也就是构造一个最小候选集，使得
$$
\sum _{y \in V_{min}^p} P(y|\hat{Y}_t,X) >= p
$$
top-p 采样根据生成概率从高到低在词表上选择累积概率恰好超过 $p$ 的候选 word 作为采样集合，再从这些候选 word 中采样出最终的结果。选出来这个集合之后也和 top-k 采样一样，重新归一化集合内 word 的概率，并把集合外词的概率设为 $0$。  

所以采样前必须先计算每个 word 对应的概率并进行排序，即要计算 Softmax，再按概率值排序。所以相比于 top-k 采样的准备工作以外，top-p 采样前还需要加一个 softmax 操作，具体逻辑在 `updateLogitsKernelWithoutLog` Kernel 中实现，直接查看源码即可，笔者就不赘述了。

获取到 `step_logits` 之后就需要进行 top-p 采样解码了，具体来说，在每个 step，首先对词表中每个 word 根据 Softmax 结果（也就是概率）进行排序，寻找概率值之和刚好大于等于 p 值的 word 子集，然后将这个子集中 word 的概率再归一化，最后从新的概率分布中采样。所以第一步就是要在形状为 `[batch_size, vocab_size]` 的`step_logits` 矩阵中分 `batch_size` 组对 word 进行排序。这里排序是个大工程，因为 `vocab_size` 通常会很大，笔者使用了 CUB 库中的 API 进行排序。  
```cpp
template <typename T>
void launchTopPSamplingKernel(const T *__restrict__ logits_probs, const int *__restrict__ id_vals, T *__restrict__ sorted_logits_probs,
                                int *__restrict__ sorted_id_vals, const int *__restrict__ topp_offset_buf, void *__restrict__ temp_storage,
                                size_t temp_storage_size, bool *__restrict__ finished_buf, const int *__restrict__ prompt_tokens,
                                const bool *__restrict__ prompt_tokens_mask, const int cur_pos, const int max_prompt_seq_len,
                                const int random_num, int *__restrict__ output_ids, int *__restrict__ sequence_length, const int end_id,
                                const int batch_size, const int vocab_size, const float probability_threshold, cudaStream_t stream)
{
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    cub::DeviceSegmentedRadixSort::SortPairsDescending(temp_storage,
                                                        temp_storage_size,
                                                        logits_probs,
                                                        sorted_logits_probs,
                                                        id_vals,
                                                        sorted_id_vals,
                                                        vocab_size * batch_size,
                                                        batch_size,
                                                        topp_offset_buf, topp_offset_buf + 1);

    int local_block_size;
    assert(batch_size <= 1024);
    if (batch_size <= 128)
    {
        local_block_size = 128;
    }
    else if (batch_size <= 256)
    {
        local_block_size = 256;
    }
    else if (batch_size <= 512)
    {
        local_block_size = 512;
    }
    else
    {
        local_block_size = 1024;
    }

    topPSampling<<<1, local_block_size, 0, stream>>>(sorted_logits_probs, sorted_id_vals, output_ids, sequence_length,
                                                        finished_buf, prompt_tokens, prompt_tokens_mask, cur_pos, max_prompt_seq_len,
                                                        batch_size, vocab_size, random_num, probability_threshold, end_id);
}
```
这里我们对 `cub::DeviceSegmentedRadixSort::SortPairsDescending` 函数的主要参数进行介绍：  
- `d_temp_storage`：设备可以访问的临时内存，当设置为 `NULL` 时，所需的分配大小将写入 `temp_storage_bytes`，并且不执行任何工作。所以在真正执行函数前，我们需要先传一下 NULL 获取 `temp_storage_bytes` 然后再开始真正的执行排序，这个获取临时空间的前置操作，笔者放在了 `DecodingSampling` 的构造函数中进行。
- `temp_storage_bytes`：临时内存缓冲区的大小。
- `d_keys_in`：排序过程中的比较依据，也就是说排序是根据这个指针指向的数据的来进行的，这里我们将它设置为概率值 `step_logits_buf_`。
- `d_keys_out`：排序后的输出，这里我们用 `sorted_logits_probs` 来接收存储。
- `d_values_in`：与 key 一一对应，这里我们把他设置为概率值对应的索引 `id_vals`，其实就是 `word_id`。
- `d_values_out`：排序后的输出，这里我们用 `sorted_id_vals` 来接收。
- `num_items`：待排序的元素数目，这里应该是 `batch_size * vocab_size`。
- `num_segments`：待排序的批次，也就是分为多少个组，这里是对每个样本单独排序，所以取 `batch_size`。
- `d_begin_offsets`：每个分组的起始索引，为了方便 `end_offset` 的设置，这个变量数组的长度通常是 `num_segments + 1`，前面 `num_segments` 个元素都是分组的起始索引，最后一个元素设为 `num_items`，这里我们设置为 `topp_offset_buf`，前面已经完成初始化。
- `d_end_offsets`：每个分组的结束索引，注意这里是“顾头不顾尾”的模式，所以直接可以设置为 `d_begin_offsets + 1`，这里我们设置为 `topp_offset_buf + 1`。

参数意义介绍完毕后，其实函数的作用也就清晰了，就是分组降序排序，每一组对应 `batch` 内的一个样本，也就是 `vocab_size` 个元素，最终我们获取到了 batch 内每个样本排序后的待采样 word 的概率值 `sorted_logits_probs`和 `sorted_id_vals`。

根据采样原理，拿到排序结果后，我们需要根据 p 值进行候选集的确定，然后在候选集的内部进行采样。笔者提供了核函数 `topPSampling` 进行采样工作，`grid_size` 设置为 `1`，`block_size` 设置为 `bacth_size`，即在一个 block 内完成计算，每个线程承担一个样本的计算任务。 
```cpp
/**
* top-k Sampling kernel
* grid(1), block(batch_size)
*/
template <typename T>
__global__ void topPSampling(const T *__restrict__ sorted_logits_probs, const int *__restrict__ sorted_id_vals,
                                int *__restrict__ ids, int *__restrict__ sequence_length, bool *__restrict__ finished_buf,
                                const int *__restrict__ prompt_tokens, const bool *__restrict__ prompt_tokens_mask,
                                const int cur_pos, const int max_prompt_seq_len, const int batch_size, const int vocab_size,
                                const int random_num, const float prob_threshold, const int end_id)
{
    if (threadIdx.x < batch_size)
    {
        // prompt phase, next_token[:] = prompt_tokens[:, cur_pos]
        if (cur_pos < max_prompt_seq_len && prompt_tokens_mask[threadIdx.x * max_prompt_seq_len + cur_pos])
        {
            ids[threadIdx.x] = prompt_tokens[threadIdx.x * max_prompt_seq_len + cur_pos];
        }
        else
        {
            int tid = threadIdx.x;
            curandState_t local_state;
            curand_init(random_num, tid, 0, &local_state);
            float rand_num = curand_uniform(&local_state) * prob_threshold;
            ids[tid] = sorted_id_vals[vocab_size - 1];

            for (int i = tid * vocab_size; i < tid * vocab_size + vocab_size; i++)
            {
                rand_num = rand_num - sorted_logits_probs[i];
                if (rand_num <= 0)
                {
                    ids[tid] = sorted_id_vals[i];
                    break;
                }
            }

            sequence_length[tid] = finished_buf[tid] ? sequence_length[tid] : sequence_length[tid] + 1;
            finished_buf[tid] = ids[tid] == end_id ? true : false;
        }
    }
}
```

首先判断当前 Decoding 阶段，如果当前样本还处于 prompt 阶段，也就是说下一个 token 是已经存在的，那直接替换即可，即 `next_token[:] = prompt_tokens[:, cur_pos]`，否则的话就需要采样解码。

采样过程和前面 top-k 的过程大同小异，有一点区别就是，不用真的先确定候选集再进行采样，可以直接一步进行。先使用 cuRand API 从均匀分布中随机一个 `0~1` 之间的数再乘以 p 值（`probability_threshold_`），这其实就相当于把采样的概率点缩放到了 p 值范围内，然后遍历 `sorted_logits_probs` 判断采样点落在哪个区间，就选中了哪个 word，示意图如下：  
![](https://mmbiz.qpic.cn/sz_mmbiz_png/GJUG0H1sS5rOU8ibicb4mGqGvrH9r0pm9fIgRJ44aurWP5bEmSedysRc2ZDsdSVoPYtEEA8c8gJX6UicBZw7gZticA/640?wx_fmt=png)

采样完成后把采样结果更新到 `ids`，然后对 `sequence_length` 和 `finished_buf_` 进行更新，至此，当前 step 的 top-p 采样解码就完成了。  

### 4.9 CheckFinished
在每次采样结束后都会更新每个样本的 `sequence_length` 和 `finished_buf_`，那么每完成一个 step 的采样后就需要对 `finished_buf_` 进行判断，如果所有的元素都为 `false`，说明 batch 内所有的样本都解码结束，这时候就需要停止解码，退出循环，否则继续下一轮解码。这里考虑到数据量不大，简单起见笔者直接把 `finished_buf_` 传输到主机端内存，在主机端进行遍历判断，这块逻辑后期还可以进行优化加速。

### 4.10 RemovePromptTokenKernel
解码停止之后，要整合一下生成的 word_id 并存储到 `decoding_params.output_ids` 中，为什么需要这一步？是因为在 prompt 计算阶段，我们为了计算方便取的是一个形状为 `batch_size, min_prompt_seq_len` 的 prompt_tokens 矩阵进行的，从而确保每个样本都在 prompt 阶段，这会导致在 generation 阶段计算时还有部分样本仍然处于 prompt 阶段，即这部分样本下一个 step 的 token 是已确定的，这部分 prompt tokens 也被存储到了 `word_ids_buf_` 中，我们要把这部分  prompt tokens 剔除，只保留生成的 tokens，举个例子。

假设 `prompt_tokens` 为如下矩阵：
```
[
    [1, 2, 3, 4, 5, 0, 0, 0],
    [1, 2, 3, 4, 0, 0, 0, 0],
    [1, 2, 3, 4, 5, 6, 0, 0],
    [1, 2, 3, 4, 5, 6, 7, 0],
]
```
相应地 `prompt_tokens_mask` 为如下矩阵：
```
[
    [1, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0],
]
```
那么生成的 `word_ids_buf_` 就会形如如下矩阵：
```
[
    [ 5, 11, 12, 13, ...],
    [21, 22, 23, 24, ...],
    [5, 6, 31, 32, ...],
    [5, 6, 7, 41, ...],
]
```
可以看出 `word_ids_buf_` 中除了第 2 行以外其他行都包含了之前 `prompt_tokens` 的元素，我们需要将其移除，调整为如下矩阵：
```
[
    [11, 12, 13, 14, ...],
    [21, 22, 23, 24, ...],
    [31, 32, 33, 34, ...],
    [41, 42, 43, 44, ...],
]
```
计算目的明确之后，计算思路也就比较简单了，同时要注意的是这里还有一个转置操作，因为 `word_ids_buf_` 内存顺序是 `[total_len, batch_size]`，而 `decoding_params.output_ids` 内存顺序是 `[batch_size, total_len]`，可以看出读操作存在“非内存合并”的问题，但是这里由于使用了 `const int *__restrict__` 修饰了 `word_ids_buf` 指针，所以编译器会考虑利用 Unified Cache 进行缓存加速，一定程度上削弱“非内存合并”读取带来的内存带宽问题，当然，追求极致性能的读者也可以考虑利用共享内存进行矩阵转置。

```cpp
__global__ void removePromptTokenKernel(int *__restrict__ gen_ids, const int *__restrict__ word_ids_buf,
                                            const int *__restrict__ sequence_length, const int *__restrict__ prompt_seq_lengths,
                                            const int min_prompt_seq_len, const int batch_size, const int total_len)
{
    const int offset = prompt_seq_lengths[blockIdx.x] - min_prompt_seq_len;
    for (int tid = threadIdx.x; tid < sequence_length[blockIdx.x]; tid += blockDim.x)
    {
        gen_ids[blockIdx.x * total_len + tid] = word_ids_buf[(offset + tid) * batch_size + blockIdx.x];
    }
}

void launchRemovePromptTokenKernel(int *__restrict__ gen_ids, const int *__restrict__ word_ids_buf, const int *__restrict__ sequence_length,
                                    const int *__restrict__ prompt_seq_lengths, const int min_prompt_seq_len, const int batch_size, const int total_len, cudaStream_t stream)
{
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    removePromptTokenKernel<<<batch_size, 256, 0, stream>>>(gen_ids, word_ids_buf, sequence_length, prompt_seq_lengths, min_prompt_seq_len, batch_size, total_len);
}
```


## 5 小结
至此，FasterLLaMA v1.0 的核心逻辑及其实现思路已经介绍完毕。现对本文总结如下：  
- 提供了一个 Decoder 模块和一套推理方案 Decoding 模型，目前仅适配 LLaMA2，至于LLaMA3 及其他开源大模型的适配工作，将在后续版本逐步加入。
- 模型量化方面，针对 GEMM 场景，提供了基于 cuBLAS 的 INT8 量化实现，可以高效地利用 GPU 中的 INT8 Tensor Core，在保证低精度损失的前提下，取得较好的加速比。
- 针对 Decoding 模型的解码场景，参考了 Faster Transformer，改写了部分逻辑，提供了两种基于采样解码的实现：top-k 解码和 top-p 解码。
- 数据类型方面，目前支持 FP32 和 FP16 两种类型，笔者针对 FP16 类型对相关 Kernel 函数模板进行了特化。
- 注意力机制方面，目前仅支持 MHA，计划在后续版本加入对 MQA 和 GQA 的支持。
- 第三方库依赖方面，除了引入 Nvidia 官方库（cuBLAS、cuRand、CUB）以外，没有任何第三方库被引入，极大地降低了用户引入集成 FasterLLaMA 代码的难度，只需要正常安装 CUDA 环境即可。
- CUDA 算子方面，提供了一系列 Transformer、量化反量化、等深度学习相关的高性能算子，并且尽可能针对具体计算任务进行了 Kernel 融合，可以供读者借鉴或引入到其他项目中。
- GEMM 运算方面，目前全部是调用 cuBLAS 库进行，后续版本考虑基于 CUTLASS 进行更细粒度的控制，以及更高层次的 Kernel 融合。
- 设备内存缓冲区方面，目前只是复用了一部分 buffer，仍然存在一定的优化空间，后续将考虑更高程度的复用，进一步节省设备内存。