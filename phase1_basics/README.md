#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段一：RAG基础理论准备
目标：掌握RAG基本概念和技术栈
"""

"""
学习目标：
1. 理解RAG（Retrieval-Augmented Generation）工作原理
2. 掌握向量化基础概念
3. 理解相似度计算方法
4. 掌握LLM API调用方法
5. 搭建开发环境

测试标准：
- 能解释RAG工作流程
- 能独立完成文档向量化
- 能计算文本相似度
- 能使用LLM生成答案
"""


# ==================== 1. RAG基础概念 ====================
"""
RAG工作流程：
1. 文档加载和预处理
2. 文档分块（Chunking）
3. 文本向量化（Embedding）
4. 向量存储
5. 用户查询向量化
6. 相似度检索
7. 上下文构建
8. LLM生成答案
9. 答案返回

RAG优势：
- 减少幻觉（Hallucination）
- 知识实时更新
- 可追溯性
- 领域知识定制

RAG挑战：
- 检索准确性
- 上下文窗口限制
- 生成质量
- 响应速度
"""


# ==================== 2. 向量化基础 ====================
"""
什么是向量化？
- 将文本转换为数值向量
- 向量表示文本的语义信息
- 相似文本在向量空间中距离接近

Embedding模型：
- OpenAI: text-embedding-3-small/large
- Sentence-Transformers: all-MiniLM-L6-v2
- BGE: bge-large-zh-v1.5
- M3E: m3e-base

向量维度：
- 通常在384-1536维之间
- 维度越高，表达能力越强
- 但计算和存储成本增加
"""


# ==================== 3. 相似度计算 ====================
"""
常用相似度计算方法：

1. 余弦相似度（Cosine Similarity）
   - 公式：cos(θ) = (A·B) / (||A|| × ||B||)
   - 范围：[-1, 1]，越接近1越相似
   - 优点：不受向量长度影响

2. 欧几里得距离（Euclidean Distance）
   - 公式：d = √(Σ(ai - bi)²)
   - 范围：[0, ∞)，越小越相似
   - 优点：直观，几何意义明确

3. 点积（Dot Product）
   - 公式：A·B = Σ(ai × bi)
   - 范围：[-∞, ∞]，越大越相似
   - 优点：计算简单快速

推荐使用：余弦相似度
"""


# ==================== 4. 开发环境配置 ====================
"""
推荐开发环境：
- Python 3.9+
- 虚拟环境：conda 或 venv
- IDE：VSCode 或 PyCharm
- GPU（可选）：NVIDIA GPU加速向量化

环境配置步骤：
1. 创建虚拟环境
2. 安装依赖：pip install -r requirements.txt
3. 配置环境变量：.env文件
4. 测试环境是否正常
"""


# ==================== 5. 实践任务清单 ====================
"""
基础任务：
□ 1.1 创建虚拟环境
□ 1.2 安装所有依赖
□ 1.3 配置环境变量（API Key等）
□ 1.4 测试OpenAI API调用
□ 1.5 实现基础文本向量化
□ 1.6 计算文本相似度
□ 1.7 理解RAG工作流程

验证任务：
□ 1.8 能独立解释RAG工作原理
□ 1.9 能完成文本向量化
□ 1.10 能计算余弦相似度
□ 1.11 能使用LLM生成答案
□ 1.12 能搭建完整RAG流程（基础版）

参考资料：
- LangChain文档: https://python.langchain.com/
- OpenAI文档: https://platform.openai.com/docs
- 向量数据库指南: https://weaviate.io/blog/what-is-a-vector-database
"""

print("阶段一：RAG基础理论准备")
print("=" * 60)
print("\n学习目标：")
print("1. 理解RAG工作原理")
print("2. 掌握向量化技术")
print("3. 理解相似度计算")
print("4. 掌握LLM API调用")
print("5. 搭建开发环境")

print("\n关键概念：")
print("- 文档分块")
print("- 文本向量化")
print("- 向量检索")
print("- 上下文构建")
print("- LLM生成")

print("\n实践任务：")
print("请参考本目录下的测试代码，完成所有基础任务")
print("验证任务通过后，即可进入下一阶段学习")
