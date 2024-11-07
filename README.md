# 使用 CUDA C++ 实现的大模型推理框架 FasterLLaMA

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