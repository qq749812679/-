# 相关论文与技术博客

本文档收集了与LLM部署相关的重要学术论文和技术博客，为深入理解各种优化技术提供参考。

## 推理优化论文

### PagedAttention与连续批处理

- **vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention**
  - 作者: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang et al. (2023)
  - 链接: [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
  - 摘要: 引入PagedAttention技术，通过非连续内存管理显著提高LLM推理效率，处理任意序列长度并提高吞吐量。

- **Efficient Memory Management for Large Language Model Serving with PagedAttention**
  - 作者: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang et al. (2023)
  - 链接: [SOSP 2023](https://dl.acm.org/doi/10.1145/3600006.3613165)
  - 摘要: 详细介绍PagedAttention的实现和评估，证明其比现有解决方案获得2-4倍吞吐量提升。

### 量化与压缩

- **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**
  - 作者: Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2023)
  - 链接: [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
  - 摘要: 提出一种后训练量化方法，能将LLM量化到4位精度，几乎不损失性能。

- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**
  - 作者: Ji Lin, Jiaming Tang, Haotian Tang et al. (2023)
  - 链接: [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
  - 摘要: 介绍一种考虑激活值重要性的权重量化方法，在INT4精度下保持性能。

- **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**
  - 作者: Guangxuan Xiao, Ji Lin, Mickael Seznec et al. (2022)
  - 链接: [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
  - 摘要: 提出通过迁移激活值困难到权重，简化量化过程，使W8A8量化达到FP16性能。

### 推理并行与分布式推理

- **FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU**
  - 作者: Ying Sheng, Lianmin Zheng, Binhang Yuan et al. (2023)
  - 链接: [arXiv:2303.06865](https://arxiv.org/abs/2303.06865)
  - 摘要: 提出一种使用计算和内存高效的推理系统，能在单个GPU上运行大型LLM。

- **DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale**
  - 作者: Reza Yazdani Aminabadi, Samyam Rajbhandari, Minjia Zhang et al. (2022)
  - 链接: [arXiv:2207.00032](https://arxiv.org/abs/2207.00032)
  - 摘要: 介绍DeepSpeed Inference，一个高效推理引擎，支持模型并行和张量并行。

- **Megablocks: Efficient Sparse Training with Mixture-of-Experts**
  - 作者: Trevor Gale, Deepak Narayanan, Cliff Young et al. (2023)
  - 链接: [arXiv:2211.15841](https://arxiv.org/abs/2211.15841)
  - 摘要: 提出一种高效稀疏训练技术，特别适用于混合专家模型的并行推理。

### 推理系统设计

- **Orca: A Distributed Serving System for Transformer-Based Generative Models**
  - 作者: Deepak Narayanan, Mohammad Shoeybi, Jared Casper et al. (2022)
  - 链接: [OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/narayanan-deepak)
  - 摘要: 介绍Orca系统，针对Transformer生成模型优化的分布式服务系统。

- **Fast Transformer Decoding: One Write-Head is All You Need**
  - 作者: Noam Shazeer (2019)
  - 链接: [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
  - 摘要: 提出一种优化Transformer解码器的方法，通过缓存中间结果加速自回归生成。

## 系统架构与部署论文

### 服务架构

- **Shepherd: Supporting Large Language Models on Heterogeneous Devices**
  - 作者: Andrew Jiang, Hasan Genc, Micah J. Smith et al. (2023)
  - 链接: [arXiv:2309.15906](https://arxiv.org/abs/2309.15906)
  - 摘要: 探讨在异构设备上支持LLM的服务架构，特别关注资源有限设备。

- **Serving Large Language Models at Scale: From Challenges to Solutions**
  - 作者: Xiake Sun, Anze Xie, Man Luo et al. (2023)
  - 链接: [arXiv:2312.15234](https://arxiv.org/abs/2312.15234)
  - 摘要: 综述大型语言模型服务化面临的挑战和解决方案，包括系统架构和优化技术。

### 缓存与内存优化

- **Efficient Large Language Model Inference with Limited Memory**
  - 作者: Yushan Su, Xuan Zhang, Jieru Mei et al. (2023)
  - 链接: [arXiv:2308.04985](https://arxiv.org/abs/2308.04985)
  - 摘要: 探讨在有限内存环境下高效运行LLM的方法，包括激活值重计算和选择性缓存。

- **MQA: Multihead Question Attention for LLM Serving**
  - 作者: Zhiyuan Zhang, Benjamin Lever, Hao He et al. (2024)
  - 链接: [arXiv:2401.15073](https://arxiv.org/abs/2401.15073)
  - 摘要: 提出一种针对LLM服务的注意力机制优化，降低内存占用提高吞吐量。

### 边缘设备部署

- **LLM in a flash: Efficient Large Language Model Inference with Limited Memory**
  - 作者: Keivan Alizadeh, Iman Mirzadeh, Mehrdad Farajtabar et al. (2023)
  - 链接: [arXiv:2312.11514](https://arxiv.org/abs/2312.11514)
  - 摘要: 探讨在闪存和有限内存环境中运行LLM的技术，适用于边缘设备。

- **SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression**
  - 作者: Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian et al. (2023)
  - 链接: [arXiv:2306.03078](https://arxiv.org/abs/2306.03078)
  - 摘要: 提出一种结合稀疏化和量化的表示方法，实现近乎无损的LLM压缩。

## 技术博客

### 系统设计

- **Deploying GPT-J: Scaling GPU Inference**
  - 作者: Hugging Face
  - 链接: [Hugging Face Blog](https://huggingface.co/blog/gptj-inference)
  - 摘要: 详细介绍部署GPT-J模型的系统设计和优化技术。

- **LLM Deployments at Scale: Practical Strategies**
  - 作者: Anyscale
  - 链接: [Anyscale Blog](https://www.anyscale.com/blog/llm-deployments-at-scale-practical-strategies)
  - 摘要: 分享大规模LLM部署的实用策略和最佳实践。

- **Serving LLMs in Distributed Systems, 10x cheaper**
  - 作者: Predibase
  - 链接: [Predibase Blog](https://predibase.com/blog/serving-llms-in-distributed-systems-10x-cheaper)
  - 摘要: 探讨如何通过分布式系统设计降低LLM服务成本。

### 推理优化

- **Engineering for LLM-Powered Chat Applications: Best Practices**
  - 作者: OpenAI
  - 链接: [OpenAI Blog](https://www.openai.com/blog/engineering-for-llm-powered-chat-applications)
  - 摘要: OpenAI分享的构建基于LLM聊天应用的工程最佳实践。

- **Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA**
  - 作者: Hugging Face
  - 链接: [Hugging Face Blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
  - 摘要: 介绍使用bitsandbytes进行4位量化和QLoRA的实际应用。

- **Optimizing LLM Performance with DeepSpeed**
  - 作者: Microsoft
  - 链接: [Microsoft DeepSpeed Blog](https://www.microsoft.com/en-us/research/blog/deepspeed-accelerating-large-scale-model-inference-and-training-via-system-optimizations-and-compression/)
  - 摘要: 微软介绍使用DeepSpeed优化LLM性能的技术。

### 监控与可观测性

- **Monitoring Production AI Systems: Beyond the Algorithms**
  - 作者: Datadog
  - 链接: [Datadog Blog](https://www.datadoghq.com/blog/monitor-ai-systems/)
  - 摘要: 讨论AI系统生产环境监控的最佳实践，超越算法层面。

- **Observability for LLM Applications**
  - 作者: New Relic
  - 链接: [New Relic Blog](https://newrelic.com/blog/best-practices/llm-observability)
  - 摘要: 探讨LLM应用的可观测性设计和实现。

### 成本优化

- **How Much Does It Cost to Train and Deploy LLMs?**
  - 作者: Scaleway
  - 链接: [Scaleway Blog](https://www.scaleway.com/en/blog/how-much-does-it-cost-to-train-and-deploy-llms/)
  - 摘要: 深入分析LLM训练和部署的成本构成及优化方法。

- **The Hidden Cost of LLM APIs and How to Fix It**
  - 作者: Modal
  - 链接: [Modal Blog](https://modal.com/blog/llm-cost-optimization)
  - 摘要: 探讨LLM API使用的隐藏成本及优化策略。

### 案例研究

- **How We Scaled Llama 2 to 100 Simultaneous Users on a Single GPU**
  - 作者: Anyscale
  - 链接: [Anyscale Blog](https://www.anyscale.com/blog/how-we-scaled-llama-2-to-100-simultaneous-users-on-a-single-gpu)
  - 摘要: 详细介绍在单GPU上支持100个并发用户使用Llama 2的技术。

- **Optimizing RAG for Production**
  - 作者: Pinecone
  - 链接: [Pinecone Blog](https://www.pinecone.io/learn/optimizing-rag-for-production/)
  - 摘要: 探讨检索增强生成(RAG)在生产环境的优化技术。

- **Building a Multi-tenant LLM Service with Ray and vLLM**
  - 作者: Anyscale
  - 链接: [Anyscale Blog](https://www.anyscale.com/blog/multi-tenant-llm-service-with-ray-and-vllm)
  - 摘要: 详细介绍使用Ray和vLLM构建多租户LLM服务的实践。

## 性能评估与基准测试

- **Language Models are Few-Shot Learners**
  - 作者: Tom B. Brown, Benjamin Mann, Nick Ryder et al. (2020)
  - 链接: [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
  - 摘要: GPT-3原始论文，探讨了大型语言模型的性能评估方法。

- **LLM Inference Unveiled: Survey and Roofline Analysis**
  - 作者: Yining Shi, Tianle Cai, Qirui Jin et al. (2023)
  - 链接: [arXiv:2402.16363](https://arxiv.org/abs/2402.16363)
  - 摘要: 对LLM推理过程进行系统性分析，提供性能评估框架。

- **Benchmarking LLMs on Serving Workloads for Cost and Latency Optimization**
  - 作者: Nikhil Sardana, Liangwei Yang, Suren Gunturu et al. (2023)
  - 链接: [arXiv:2312.05234](https://arxiv.org/abs/2312.05234)
  - 摘要: 提供LLM服务工作负载的基准测试方法，优化成本和延迟。 