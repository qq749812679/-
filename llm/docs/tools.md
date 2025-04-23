# 推荐工具与框架

本文档汇总了LLM部署过程中常用的工具、框架和库，帮助团队选择合适的技术栈。

## 模型推理框架

| 框架名称 | 主要特点 | 适用场景 | 链接 |
|---------|---------|---------|------|
| **vLLM** | 高性能PagedAttention、连续批处理 | 高吞吐量生产环境 | [vLLM](https://github.com/vllm-project/vllm) |
| **TGI** | HuggingFace推出的文本生成推理 | 与HF生态系统集成 | [Text Generation Inference](https://github.com/huggingface/text-generation-inference) |
| **LLaMA.cpp** | 高效CPU推理、量化支持 | 资源受限环境 | [LLaMA.cpp](https://github.com/ggerganov/llama.cpp) |
| **CTransformers** | 基于GGML的C推理库 | 轻量级部署 | [CTransformers](https://github.com/marella/ctransformers) |
| **TensorRT-LLM** | NVIDIA优化的GPU推理 | NVIDIA GPU部署 | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |
| **DeepSpeed Inference** | 分布式大模型推理 | 多GPU/多节点部署 | [DeepSpeed](https://github.com/microsoft/DeepSpeed) |
| **ONNX Runtime** | 跨平台模型推理优化 | 通用模型部署 | [ONNX Runtime](https://github.com/microsoft/onnxruntime) |

## 量化工具

| 工具名称 | 支持位宽 | 特点 | 链接 |
|---------|---------|------|------|
| **GPTQ** | INT4/INT8 | 权重量化，精度损失小 | [GPTQ](https://github.com/IST-DASLab/gptq) |
| **AWQ** | INT4/INT8 | 激活感知量化 | [AWQ](https://github.com/mit-han-lab/llm-awq) |
| **SqueezeLLM** | 2-8位 | 低位宽量化 | [SqueezeLLM](https://github.com/squeezeailab/SqueezeLLM) |
| **bitsandbytes** | INT8/FP8 | HF集成良好 | [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) |
| **llama.cpp** | 2-8位 | 内置多种量化方法 | [llama.cpp](https://github.com/ggerganov/llama.cpp) |

## 服务托管与编排

| 工具名称 | 类型 | 主要功能 | 适用场景 |
|---------|------|---------|---------|
| **Kubernetes** | 容器编排 | 大规模容器管理、自动扩缩容 | 生产级部署 |
| **Docker** | 容器化 | 环境隔离、一致性部署 | 开发和生产环境 |
| **Helm** | K8s包管理 | LLM服务模板和配置管理 | Kubernetes部署 |
| **Ray Serve** | 服务框架 | 分布式模型服务 | ML专用服务 |
| **BentoML** | ML服务平台 | 模型打包、部署、监控 | 端到端ML服务 |
| **Seldon Core** | K8s上的ML部署 | 复杂ML管道部署 | 企业级ML服务 |
| **KServe** | K8s上的模型服务 | 无服务器推理 | 云原生模型部署 |

## API网关与负载均衡

| 工具名称 | 主要功能 | 特点 | 链接 |
|---------|---------|------|------|
| **Kong** | API网关 | 插件生态、高性能 | [Kong](https://github.com/Kong/kong) |
| **Tyk** | API网关 | 开源、易部署 | [Tyk](https://github.com/TykTechnologies/tyk) |
| **Traefik** | 边缘路由器 | 自动服务发现、易配置 | [Traefik](https://github.com/traefik/traefik) |
| **Nginx** | Web服务器/反向代理 | 高性能、稳定性好 | [Nginx](https://nginx.org/) |
| **HAProxy** | 负载均衡器 | TCP/HTTP负载均衡 | [HAProxy](http://www.haproxy.org/) |
| **Envoy** | 服务代理 | 现代化负载均衡 | [Envoy](https://github.com/envoyproxy/envoy) |

## 监控与可观测性

| 工具名称 | 类型 | 主要功能 | 链接 |
|---------|------|---------|------|
| **Prometheus** | 监控系统 | 时间序列数据收集和告警 | [Prometheus](https://prometheus.io/) |
| **Grafana** | 可视化平台 | 指标数据可视化与仪表盘 | [Grafana](https://grafana.com/) |
| **Jaeger** | 分布式追踪 | 请求路径和性能追踪 | [Jaeger](https://www.jaegertracing.io/) |
| **OpenTelemetry** | 可观测性框架 | 指标、日志、追踪集成 | [OpenTelemetry](https://opentelemetry.io/) |
| **Datadog** | 商业监控平台 | 全栈监控与APM | [Datadog](https://www.datadoghq.com/) |
| **New Relic** | 商业APM | 应用性能监控 | [New Relic](https://newrelic.com/) |
| **DynaTrace** | 商业监控 | AI辅助监控 | [DynaTrace](https://www.dynatrace.com/) |

## 向量数据库与检索

| 数据库名称 | 特点 | 适用场景 | 链接 |
|-----------|------|---------|------|
| **Pinecone** | 托管服务、高扩展性 | 生产级RAG | [Pinecone](https://www.pinecone.io/) |
| **Weaviate** | 开源、模块化 | 多模态检索 | [Weaviate](https://github.com/weaviate/weaviate) |
| **Milvus** | 分布式向量数据库 | 大规模向量检索 | [Milvus](https://github.com/milvus-io/milvus) |
| **Qdrant** | 向量相似度搜索 | 精确过滤与搜索 | [Qdrant](https://github.com/qdrant/qdrant) |
| **Chroma** | 轻量级嵌入式数据库 | 开发与小型部署 | [Chroma](https://github.com/chroma-core/chroma) |
| **FAISS** | Facebook的相似性搜索库 | 高性能向量搜索 | [FAISS](https://github.com/facebookresearch/faiss) |
| **Vespa** | 搜索引擎和向量数据库 | 生产级搜索应用 | [Vespa](https://github.com/vespa-engine/vespa) |

## LLM应用框架

| 框架名称 | 主要功能 | 特点 | 链接 |
|---------|---------|------|------|
| **LangChain** | LLM应用构建框架 | 链式处理、集成多种工具 | [LangChain](https://github.com/langchain-ai/langchain) |
| **LlamaIndex** | 数据接入与RAG框架 | 数据连接与检索增强 | [LlamaIndex](https://github.com/jerryjliu/llama_index) |
| **Haystack** | NLP与LLM管道框架 | 模块化组件、灵活管道 | [Haystack](https://github.com/deepset-ai/haystack) |
| **FlowiseAI** | 可视化LLM应用构建 | 拖放式界面 | [FlowiseAI](https://github.com/FlowiseAI/Flowise) |
| **LangFlow** | 可视化LangChain构建工具 | 用户友好界面 | [LangFlow](https://github.com/logspace-ai/langflow) |
| **Guidance** | 结构化LLM输出控制 | 精确控制生成 | [Guidance](https://github.com/microsoft/guidance) |
| **DSPy** | 编程LLM框架 | 模块化提示工程 | [DSPy](https://github.com/stanfordnlp/dspy) |

## 缓存与队列工具

| 工具名称 | 类型 | 主要用途 | 链接 |
|---------|------|---------|------|
| **Redis** | 内存数据存储 | 响应缓存、KV缓存 | [Redis](https://github.com/redis/redis) |
| **Memcached** | 内存缓存系统 | 高性能缓存 | [Memcached](https://github.com/memcached/memcached) |
| **RabbitMQ** | 消息队列 | 请求队列管理 | [RabbitMQ](https://github.com/rabbitmq/rabbitmq-server) |
| **Kafka** | 分布式事件流平台 | 高吞吐量请求处理 | [Kafka](https://github.com/apache/kafka) |
| **Celery** | 分布式任务队列 | 异步任务处理 | [Celery](https://github.com/celery/celery) |

## 开发与调试工具

| 工具名称 | 用途 | 特点 | 链接 |
|---------|------|------|------|
| **Weights & Biases** | 实验跟踪与可视化 | 模型性能监控 | [W&B](https://wandb.ai/) |
| **TensorBoard** | 可视化工具 | TensorFlow生态集成 | [TensorBoard](https://www.tensorflow.org/tensorboard) |
| **MLflow** | ML生命周期管理 | 实验跟踪、模型注册 | [MLflow](https://github.com/mlflow/mlflow) |
| **Streamlit** | 交互式应用构建 | 快速原型开发 | [Streamlit](https://github.com/streamlit/streamlit) |
| **Gradio** | ML模型演示界面 | 易用的UI生成 | [Gradio](https://github.com/gradio-app/gradio) |
| **LMFlow** | LLM微调和评估 | 微调工作流管理 | [LMFlow](https://github.com/OptimalScale/LMFlow) |

## 性能测试工具

| 工具名称 | 主要功能 | 特点 | 链接 |
|---------|---------|------|------|
| **Locust** | 分布式负载测试 | Python编写、可扩展 | [Locust](https://github.com/locustio/locust) |
| **JMeter** | 性能和负载测试 | 成熟的测试工具 | [JMeter](https://jmeter.apache.org/) |
| **k6** | 现代负载测试工具 | 开发友好、云集成 | [k6](https://github.com/grafana/k6) |
| **wrk** | HTTP基准测试工具 | 高性能、轻量级 | [wrk](https://github.com/wg/wrk) |
| **Hey** | HTTP负载生成器 | 简单易用 | [Hey](https://github.com/rakyll/hey) |

## 安全工具

| 工具名称 | 主要功能 | 特点 | 链接 |
|---------|---------|------|------|
| **Vault** | 秘密管理 | API密钥和凭证保护 | [Vault](https://github.com/hashicorp/vault) |
| **OPA** | 策略执行 | 细粒度访问控制 | [OPA](https://github.com/open-policy-agent/opa) |
| **Trivy** | 容器漏洞扫描 | 全面的漏洞扫描 | [Trivy](https://github.com/aquasecurity/trivy) |
| **ModSecurity** | Web应用防火墙 | HTTP流量过滤 | [ModSecurity](https://github.com/SpiderLabs/ModSecurity) |
| **Falco** | 运行时安全监控 | 异常检测 | [Falco](https://github.com/falcosecurity/falco) |

## 基础设施即代码(IaC)工具

| 工具名称 | 主要功能 | 特点 | 链接 |
|---------|---------|------|------|
| **Terraform** | 基础设施配置 | 多云支持、声明式 | [Terraform](https://github.com/hashicorp/terraform) |
| **Pulumi** | 基础设施即代码 | 使用编程语言 | [Pulumi](https://github.com/pulumi/pulumi) |
| **AWS CDK** | AWS资源定义 | 使用编程语言 | [AWS CDK](https://github.com/aws/aws-cdk) |
| **Ansible** | 配置管理 | 无代理架构 | [Ansible](https://github.com/ansible/ansible) |
| **Kustomize** | Kubernetes配置 | 无模板配置管理 | [Kustomize](https://github.com/kubernetes-sigs/kustomize) |

## 云平台LLM服务

| 服务名称 | 提供商 | 特点 | 链接 |
|---------|--------|------|------|
| **Azure OpenAI** | Microsoft | OpenAI模型托管版本 | [Azure OpenAI](https://azure.microsoft.com/services/cognitive-services/openai-service/) |
| **AWS Bedrock** | Amazon | 多种基础模型API | [AWS Bedrock](https://aws.amazon.com/bedrock/) |
| **Vertex AI** | Google | Google模型与自定义模型 | [Vertex AI](https://cloud.google.com/vertex-ai) |
| **OpenAI API** | OpenAI | GPT模型家族 | [OpenAI API](https://openai.com/api/) |
| **Anthropic Claude** | Anthropic | Claude模型系列 | [Anthropic API](https://www.anthropic.com/product) |
| **AI Studio** | Hugging Face | 模型微调和部署 | [HF AI Studio](https://huggingface.co/spaces) | 