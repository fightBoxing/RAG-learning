# RAG技术学习与测试计划

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装核心依赖
pip install -r requirements_core.txt
```

### 2. 配置API密钥

本项目支持多种LLM API，包括OpenAI、智谱AI等。

#### 使用智谱AI（推荐）
编辑 `.env` 文件：
```bash
OPENAI_API_KEY=your_zhipu_api_key
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
OPENAI_MODEL=glm-4-flash
```

#### 使用OpenAI
编辑 `.env` 文件：
```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo
```

### 3. 测试配置

```bash
cd phase1_basics
python test_zhipu_ai.py  # 测试智谱AI
python test_rag_basics.py  # 测试基础功能
```

**详细配置说明请查看 [INSTALL.md](INSTALL.md)**

## 一、RAG技术核心知识点

### 1. 数据处理层
- 文档解析技术（PDF、HTML、Markdown、Word等）
- 文本清洗和标准化
- 文档分块策略（固定大小、语义分割、基于结构）
- 元数据管理

### 2. 向量化层
- Embedding模型选择（OpenAI、Cohere、HuggingFace、Sentence-Transformers）
- 向量维度选择与性能权衡
- 批量向量化优化
- 多语言Embedding

### 3. 向量数据库
- 向量数据库选型（Chroma、FAISS、Pinecone、Milvus、Weaviate）
- 向量索引算法（HNSW、IVF、Annoy）
- 相似度计算方法（余弦相似度、欧几里得距离、点积）
- 向量数据库性能优化

### 4. 检索策略
- 基础检索（Top-K检索）
- 混合检索（关键词+向量检索）
- 查询扩展和改写（Query Expansion）
- 重排序策略（Cross-Encoder、Rerankers）
- 上下文窗口管理

### 5. 提示工程
- 上下文构建策略
- Few-shot学习示例设计
- 指令工程
- 多轮对话上下文管理

### 6. 生成优化
- 温度参数控制
- 输出格式约束
- 流式响应处理
- 答案溯源

### 7. 评估体系
- 检索质量评估（Precision@K、Recall@K、MRR、NDCG）
- 生成质量评估（BLEU、ROUGE、BERTScore）
- 端到端评估（RAGAS、TruLens）
- 人工评估标准

## 二、学习阶段规划

### 阶段一：基础理论准备（第1周）

**目标**: 掌握RAG基本概念和技术栈

**学习内容**:
1. RAG原理和架构理解
2. 向量化技术基础
3. 相似度计算原理
4. LLM基础和API调用

**实践任务**:
- [ ] 搭建开发环境
- [ ] 使用OpenAI或本地LLM完成简单问答
- [ ] 实现基础文本向量化
- [ ] 理解余弦相似度计算

**测试标准**:
- 能解释RAG工作流程
- 能独立完成文档向量化
- 能计算文本相似度

### 阶段二：数据处理与向量化（第2-3周）

**目标**: 掌握高质量数据处理和向量化方法

**学习内容**:
1. 多格式文档解析
2. 文本清洗最佳实践
3. 文档分块策略对比
4. Embedding模型选择和优化
5. 向量数据库搭建

**实践任务**:
- [ ] 实现PDF/HTML/Markdown解析
- [ ] 实现多种分块策略
- [ ] 测试不同Embedding模型性能
- [ ] 搭建向量数据库并完成CRUD
- [ ] 实现批量向量化

**测试标准**:
- 能解析多种文档格式
- 分块质量提升30%以上
- 向量化准确率>95%
- 向量数据库响应时间<100ms

### 阶段三：检索策略优化（第4-5周）

**目标**: 提升检索准确率和召回率

**学习内容**:
1. Top-K检索优化
2. 混合检索实现
3. 查询扩展技术
4. 重排序算法
5. 上下文窗口管理

**实践任务**:
- [ ] 实现基础Top-K检索
- [ ] 实现混合检索（BM25+向量）
- [ ] 实现查询重写和扩展
- [ ] 集成Cross-Encoder重排序
- [ ] 实现动态上下文窗口

**测试标准**:
- 检索准确率提升20%
- 召回率提升15%
- 检索响应时间<50ms
- 能处理复杂查询

### 阶段四：RAG系统集成（第6-7周）

**目标**: 构建完整的RAG系统

**学习内容**:
1. 提示工程最佳实践
2. 上下文构建策略
3. 生成参数优化
4. 多轮对话支持
5. 答案溯源

**实践任务**:
- [ ] 设计高效Prompt模板
- [ ] 实现上下文动态选择
- [ ] 优化生成参数
- [ ] 实现多轮对话
- [ ] 添加答案溯源功能

**测试标准**:
- 端到端响应准确率>80%
- 答案相关性>85%
- 多轮对话连贯性良好
- 能溯源答案来源

### 阶段五：评估与优化（第8周）

**目标**: 建立评估体系并持续优化

**学习内容**:
1. 检索评估指标
2. 生成评估指标
3. 端到端评估工具
4. A/B测试方法
5. 性能监控和优化

**实践任务**:
- [ ] 实现检索质量评估
- [ ] 实现生成质量评估
- [ ] 集成RAGAS评估框架
- [ ] 建立监控Dashboard
- [ ] 进行A/B测试对比

**测试标准**:
- 建立完整评估体系
- 自动化评估覆盖>80%
- 性能指标可视化
- 持续优化机制

### 阶段六：高级优化（第9-10周）

**目标**: 掌握高级优化技术

**学习内容**:
1. 知识图谱增强
2. Agent模式应用
3. 微调Embedding模型
4. 多模态RAG
5. 实时更新机制

**实践任务**:
- [ ] 探索知识图谱集成
- [ ] 实现Agent模式RAG
- [ ] 微调领域特定Embedding
- [ ] 支持图片/表格等多模态
- [ ] 实现实时知识更新

**测试标准**:
- 特定领域准确率提升>10%
- 支持复杂推理任务
- 系统稳定性>99%

## 三、项目实战项目

### 项目1: 基础RAG问答系统
- 功能: 基于文档的问答
- 技术点: 文档解析、向量化、基础检索
- 难度: ⭐⭐

### 项目2: 企业知识库搜索
- 功能: 企业内部文档智能搜索
- 技术点: 混合检索、重排序、答案溯源
- 难度: ⭐⭐⭐

### 项目3: 多轮对话RAG助手
- 功能: 支持上下文的对话助手
- 技术点: 对话管理、上下文窗口、提示工程
- 难度: ⭐⭐⭐⭐

### 项目4: 领域专家RAG系统
- 功能: 特定领域的专业问答
- 技术点: 领域Embedding、知识图谱、微调
- 难度: ⭐⭐⭐⭐⭐

## 四、关键技术对比

### Embedding模型对比
| 模型 | 维度 | 性能 | 速度 | 语言支持 | 推荐场景 |
|------|------|------|------|----------|----------|
| OpenAI text-embedding-3 | 1536/3072 | 高 | 中 | 多语言 | 通用场景 |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 中高 | 快 | 多语言 | 轻量级应用 |
| bge-large-en | 1024 | 高 | 中 | 英文/中文 | 高精度需求 |
| m3e-base | 768 | 中 | 快 | 中文 | 中文场景 |

### 向量数据库对比
| 数据库 | 开源 | 性能 | 易用性 | 特性 | 推荐场景 |
|--------|------|------|--------|------|----------|
| Chroma | ✅ | 中 | 高 | 简单易用 | 快速原型 |
| FAISS | ✅ | 高 | 中 | 高性能 | 大规模检索 |
| Pinecone | ❌ | 高 | 高 | 云服务 | 生产环境 |
| Milvus | ✅ | 高 | 中 | 功能丰富 | 企业级应用 |

### 检索策略对比
| 策略 | 准确率 | 速度 | 复杂度 | 适用场景 |
|------|--------|------|--------|----------|
| 纯向量检索 | 中 | 快 | 低 | 通用场景 |
| BM25 | 中 | 快 | 低 | 关键词匹配 |
| 混合检索 | 高 | 中 | 中 | 高精度需求 |
| 重排序 | 很高 | 慢 | 高 | 精确度优先 |

## 五、性能优化清单

### 数据处理优化
- [ ] 并行文档解析
- [ ] 增量更新机制
- [ ] 文档去重和去噪
- [ ] 元数据索引优化

### 向量化优化
- [ ] 批量向量化
- [ ] GPU加速
- [ ] 模型量化
- [ ] 向量缓存

### 检索优化
- [ ] 索引优化（HNSW参数调优）
- [ ] 缓存热门查询
- [ ] 预计算策略
- [ ] 负载均衡

### 生成优化
- [ ] Prompt模板优化
- [ ] 上下文压缩
- [ ] 流式响应
- [ ] 结果缓存

## 六、常见问题与解决方案

### 问题1: 检索结果不相关
**解决方案**:
- 优化文档分块策略
- 尝试混合检索
- 使用查询扩展
- 应用重排序

### 问题2: 答案不准确
**解决方案**:
- 提高Top-K数量
- 优化Prompt设计
- 使用Few-shot示例
- 微调领域模型

### 问题3: 响应速度慢
**解决方案**:
- 优化向量索引
- 实施缓存策略
- 使用轻量模型
- 并行处理

### 问题4: 上下文丢失
**解决方案**:
- 动态上下文窗口
- 关键信息优先
- 多轮对话管理
- 记忆机制

## 七、学习资源推荐

### 官方文档
- LangChain: https://python.langchain.com/
- LlamaIndex: https://docs.llamaindex.ai/
- Chroma: https://docs.trychroma.com/
- FAISS: https://faiss.ai/

### 论文
- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- Dense Passage Retrieval for Open-Domain Question Answering
- ColBERT: Late Interaction via ColBERT

### 开源项目
- LangChain
- LlamaIndex
- Haystack
- RAGatouille

### 课程
- DeepLearning.AI: LangChain for LLM Application Development
- Coursera: Generative AI with Large Language Models

## 八、评估指标详细说明

### 检索指标
- **Precision@K**: Top-K结果中相关文档的比例
- **Recall@K**: 相关文档在Top-K中的覆盖率
- **MRR (Mean Reciprocal Rank)**: 第一个相关文档的排名倒数的平均值
- **NDCG (Normalized Discounted Cumulative Gain)**: 考虑排序质量的归一化指标

### 生成指标
- **BLEU**: 与参考答案的n-gram重叠率
- **ROUGE**: 摘要质量评估
- **BERTScore**: 基于BERT的语义相似度
- **Faithfulness**: 答案与检索上下文的一致性

### 端到端指标
- **Answer Relevancy**: 答案与问题的相关性
- **Context Precision**: 检索上下文的精确度
- **Context Recall**: 检索上下文的召回率
- **Context Entity Recall**: 实体级别的召回率

## 九、技术栈推荐

### Python基础
- Python 3.9+
- 虚拟环境管理 (conda/venv)

### 核心库
- `langchain`: RAG框架
- `llama-index`: RAG框架
- `chromadb`: 向量数据库
- `sentence-transformers`: Embedding模型
- `openai`: LLM API
- `tiktoken`: Token计数

### 辅助库
- `pypdf`: PDF解析
- `beautifulsoup4`: HTML解析
- `python-docx`: Word解析
- `numpy`: 数值计算
- `pandas`: 数据处理
- `fastapi`: API服务

### 可选库
- `faiss-cpu`: FAISS向量索引
- `rank-bm25`: BM25算法
- `cohere`: 重排序模型
- `ragas`: 评估框架
- `trulens`: 评估工具

## 十、学习检查清单

### 基础能力
- [ ] 理解RAG原理
- [ ] 掌握Python基础
- [ ] 了解向量空间概念
- [ ] 会使用Git版本控制

### 核心技能
- [ ] 能解析多种文档格式
- [ ] 能实现文本向量化
- [ ] 能搭建向量数据库
- [ ] 能实现检索逻辑
- [ ] 能设计高效Prompt
- [ ] 能集成LLM生成答案

### 高级能力
- [ ] 能优化检索策略
- [ ] 能设计评估体系
- [ ] 能进行性能调优
- [ ] 能处理大规模数据
- [ ] 能支持多模态
- [ ] 能实现实时更新

## 十一、持续学习路径

### 短期目标（1-3个月）
- 完成基础RAG系统
- 掌握核心技术栈
- 完成1-2个实战项目

### 中期目标（3-6个月）
- 深入优化检索策略
- 掌握评估体系
- 完成复杂项目

### 长期目标（6-12个月）
- 掌握高级优化技术
- 能应对生产环境挑战
- 形成自己的技术体系

---

## 总结

RAG技术是一个多学科交叉的领域，需要掌握数据处理、向量化、检索、生成等多个方面的知识。本学习计划按照从基础到高级、从理论到实践的顺序设计，帮助你系统性地掌握RAG技术。

**关键要点**:
1. 数据质量是RAG的基础
2. 向量化质量决定检索上限
3. 检索策略影响整体性能
4. 提示工程优化生成效果
5. 评估体系保障持续优化
