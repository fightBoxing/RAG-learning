#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG基础测试 - 验证环境配置和基础概念
"""

import os
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def test_environment():
    """测试环境配置"""
    print("=" * 60)
    print("测试1：环境配置")
    print("=" * 60)

    # 检查Python版本
    import sys
    print(f"Python版本: {sys.version}")

    # 检查关键库
    try:
        import openai
        print(f"✓ OpenAI版本: {openai.__version__}")
    except ImportError:
        print("✗ OpenAI未安装，请运行: pip install openai")

    try:
        import langchain
        print(f"✓ LangChain版本: {langchain.__version__}")
    except ImportError:
        print("✗ LangChain未安装，请运行: pip install langchain")

    try:
        import chromadb
        print(f"✓ ChromaDB版本: {chromadb.__version__}")
    except ImportError:
        print("✗ ChromaDB未安装，请运行: pip install chromadb")

    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠ 警告：未设置OPENAI_API_KEY环境变量")
        print("请在.env文件中添加: OPENAI_API_KEY=your_api_key")
    else:
        print(f"✓ OPENAI_API_KEY已设置")

    print("\n")


def test_text_embedding():
    """测试文本向量化"""
    print("=" * 60)
    print("测试2：文本向量化")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer

        # 加载预训练模型 embedding model
        print("加载模型：all-MiniLM-L6-v2")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # 测试文本 这里可以传入多个文本 每个文本都是一个独立的样本
        texts = [
            "RAG是一种结合检索和生成的AI技术",
            "检索增强生成可以提高答案准确性",
            "今天天气很好"
        ]

        # 生成向量
        embeddings = model.encode(texts,chunk_size=512)

        print(f"\n文本数量: {len(texts)}")
        print(f"向量维度: {embeddings.shape[1]}")
        print(f"第一个向量形状: {embeddings[0].shape}")
        print(f"向量范围: [{embeddings.min():.4f}, {embeddings.max():.4f}]")

        print("\n✓ 文本向量化成功")
        return embeddings, model

    except ImportError:
        print("✗ sentence-transformers未安装")
        print("请运行: pip install sentence-transformers")
        return None, None

    print("\n")


def test_similarity_computation(embeddings: np.ndarray):
    """测试相似度计算"""
    print("=" * 60)
    print("测试3：相似度计算")
    print("=" * 60)

    if embeddings is None:
        print("✗ 需要先完成文本向量化测试")
        return

    # 计算余弦相似度
    similarity_matrix = cosine_similarity(embeddings)

    print("余弦相似度矩阵:")
    print("-" * 60)
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix[i])):
            print(f"{similarity_matrix[i][j]:.4f} ", end="")
        print()

    # 解读相似度
    print("\n相似度解读:")
    print(f"- 文本0与文本1相似度: {similarity_matrix[0][1]:.4f} (相关)")
    print(f"- 文本0与文本2相似度: {similarity_matrix[0][2]:.4f} (不相关)")
    print(f"- 文本1与文本2相似度: {similarity_matrix[1][2]:.4f} (不相关)")

    print("\n✓ 相似度计算成功")
    print("\n")


def test_llm_generation():
    """测试LLM生成"""
    print("=" * 60)
    print("测试4：LLM生成")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠ 跳过测试：未设置OPENAI_API_KEY")
        print("\n")
        return

    try:
        from openai import OpenAI

        # 从环境变量读取配置
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

        print(f"使用模型: {model}")
        print(f"API地址: {base_url}")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        # 测试简单的问答
        question = "什么是RAG技术？"
        print(f"\n问题: {question}")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个乐于助人的AI助手。"},
                {"role": "user", "content": question}
            ],
            temperature=0.3,
            max_tokens=200
        )

        answer = response.choices[0].message.content
        print(f"回答: {answer}")

        print("\n✓ LLM生成成功")

    except Exception as e:
        print(f"✗ LLM生成失败: {e}")

    print("\n")


def test_simple_rag_pipeline():
    """测试简单的RAG流程"""
    print("=" * 60)
    print("测试5：简单RAG流程")
    print("=" * 60)

    # 知识库
    knowledge_base = [
        "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI技术",
        "RAG可以减少大模型的幻觉问题",
        "RAG的工作流程包括文档处理、向量化、检索、生成四个步骤"
    ]

    # 用户查询
    query = "RAG有什么优势？"

    print(f"知识库: {len(knowledge_base)}条")
    for i, doc in enumerate(knowledge_base):
        print(f"  {i + 1}. {doc}")

    print(f"\n用户查询: {query}")

    # 加载模型
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # 向量化知识库和查询
        doc_embeddings = model.encode(knowledge_base)
        query_embedding = model.encode([query])

        # 检索相关文档
        similarity_matrix = cosine_similarity(query_embedding, doc_embeddings)
        top_k = 1
        top_indices = np.argsort(similarity_matrix[0])[::-1][:top_k]

        print(f"\n检索到最相关的文档:")
        for idx in top_indices:
            print(f"  - {knowledge_base[idx]} (相似度: {similarity_matrix[0][idx]:.4f})")

        print("\n✓ 简单RAG流程测试成功")

    except Exception as e:
        print(f"✗ RAG流程测试失败: {e}")

    print("\n")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("RAG基础测试套件")
    print("=" * 60 + "\n")

    # 测试1：环境配置
    test_environment()

    # 测试2：文本向量化
    embeddings, model = test_text_embedding()

    # 测试3：相似度计算
    if embeddings is not None:
        test_similarity_computation(embeddings)


    # 测试4：LLM生成
    test_llm_generation()

    # 测试5：简单RAG流程
    test_simple_rag_pipeline()

    print("=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n如果所有测试都通过，恭喜你完成了阶段一的基础学习！")
    print("下一阶段：数据处理与向量化优化")


if __name__ == "__main__":
    main()
