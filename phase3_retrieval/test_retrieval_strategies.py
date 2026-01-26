#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索策略测试 - 验证检索优化功能
"""

import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity


def test_top_k_retrieval():
    """测试Top-K检索"""
    print("=" * 60)
    print("测试1：Top-K检索")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer
        import chromadb

        # 加载模型
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # 创建向量数据库
        client = chromadb.Client()
        collection = client.create_collection(name="test_topk")

        # 测试文档
        documents = [
            "RAG是检索增强生成技术",
            "向量数据库用于存储文本向量",
            "Embedding模型将文本转换为向量",
            "LLM大语言模型可以生成文本",
            "Python是一种编程语言"
        ]

        # 向量化并添加
        embeddings = model.encode(documents)
        collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            ids=[f"doc{i}" for i in range(len(documents))]
        )

        print("✓ 创建测试数据库成功")
        print(f"文档数量: {len(documents)}")

        # 测试查询
        query = "什么是RAG技术？"
        print(f"\n查询: {query}")

        # 向量化查询
        query_embedding = model.encode([query])

        # 检索
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=3
        )

        print("\nTop-3检索结果:")
        print("-" * 60)
        for i, (doc, distance) in enumerate(zip(
            results['documents'][0],
            results['distances'][0]
        )):
            similarity = 1 - distance  # Cosine距离转相似度
            print(f"{i + 1}. {doc}")
            print(f"   相似度: {similarity:.4f}")

        print("\n✓ Top-K检索测试成功")

    except Exception as e:
        print(f"✗ 测试失败: {e}")

    print()


def test_bm25_retrieval():
    """测试BM25检索"""
    print("=" * 60)
    print("测试2：BM25检索")
    print("=" * 60)

    try:
        from rank_bm25 import BM25Okapi

        # 测试文档
        documents = [
            ["RAG", "是", "检索", "增强", "生成", "技术"],
            ["向量", "数据库", "用于", "存储", "文本", "向量"],
            ["Embedding", "模型", "将", "文本", "转换", "为", "向量"],
            ["LLM", "大", "语言", "模型", "可以", "生成", "文本"],
            ["Python", "是", "一种", "编程", "语言"]
        ]

        # 构建BM25索引
        bm25 = BM25Okapi(documents)

        print("✓ BM25索引构建成功")
        print(f"文档数量: {len(documents)}")

        # 测试查询
        query = ["RAG", "技术"]
        print(f"\n查询: {' '.join(query)}")

        # 检索
        scores = bm25.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:3]

        print("\nTop-3检索结果:")
        print("-" * 60)
        for rank, idx in enumerate(top_indices):
            doc_text = ''.join(documents[idx])
            score = scores[idx]
            print(f"{rank + 1}. {doc_text}")
            print(f"   分数: {score:.4f}")

        print("\n✓ BM25检索测试成功")

    except Exception as e:
        print(f"✗ 测试失败: {e}")

    print()


def test_hybrid_retrieval():
    """测试混合检索"""
    print("=" * 60)
    print("测试3：混合检索（向量+BM25）")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer
        from rank_bm25 import BM25Okapi

        # 加载模型
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # 测试文档
        documents = [
            "RAG是检索增强生成技术",
            "向量数据库用于存储文本向量",
            "Embedding模型将文本转换为向量",
            "LLM大语言模型可以生成文本",
            "Python是一种编程语言"
        ]

        # 向量化
        doc_embeddings = model.encode(documents)

        # BM25索引
        tokenized_docs = [doc.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)

        print("✓ 混合索引构建成功")
        print(f"文档数量: {len(documents)}")

        # 测试查询
        query = "什么是RAG技术？"
        query_tokens = query.split()
        print(f"\n查询: {query}")

        # 向量检索
        query_embedding = model.encode([query])
        vector_scores = cosine_similarity(query_embedding, doc_embeddings)[0]

        # BM25检索
        bm25_scores = bm25.get_scores(query_tokens)

        # 归一化
        vector_scores_norm = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min() + 1e-8)
        bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)

        # 混合分数（加权融合）
        w_vector = 0.6
        w_bm25 = 0.4
        hybrid_scores = w_vector * vector_scores_norm + w_bm25 * bm25_scores_norm

        # Top-K
        top_indices = np.argsort(hybrid_scores)[::-1][:3]

        print("\n混合检索结果 (Top-3):")
        print("-" * 60)
        for rank, idx in enumerate(top_indices):
            doc = documents[idx]
            v_score = vector_scores[idx]
            b_score = bm25_scores[idx]
            h_score = hybrid_scores[idx]
            print(f"{rank + 1}. {doc}")
            print(f"   向量分数: {v_score:.4f}, BM25分数: {b_score:.4f}")
            print(f"   混合分数: {h_score:.4f}")

        print("\n✓ 混合检索测试成功")

    except Exception as e:
        print(f"✗ 测试失败: {e}")

    print()


def test_query_expansion():
    """测试查询扩展"""
    print("=" * 60)
    print("测试4：查询扩展")
    print("=" * 60)

    # 原始查询
    original_query = "RAG技术"

    # 同义词扩展
    synonyms = {
        "RAG": ["检索增强生成", "Retrieval-Augmented Generation"],
        "技术": ["方法", "技术方案", "解决方案"]
    }

    def expand_query(query: str, synonyms: dict) -> List[str]:
        """查询扩展"""
        expanded_queries = [query]

        words = query.split()
        for word in words:
            if word in synonyms:
                for syn in synonyms[word]:
                    expanded = query.replace(word, syn)
                    expanded_queries.append(expanded)

        return list(set(expanded_queries))  # 去重

    expanded_queries = expand_query(original_query, synonyms)

    print("原始查询:")
    print(f"  {original_query}")

    print("\n扩展查询:")
    print("-" * 60)
    for i, query in enumerate(expanded_queries):
        print(f"{i + 1}. {query}")

    print(f"\n✓ 查询扩展成功，共{len(expanded_queries)}个查询")

    print()


def test_context_window():
    """测试上下文窗口管理"""
    print("=" * 60)
    print("测试5：上下文窗口管理")
    print("=" * 60)

    # 模拟检索结果
    retrieved_docs = [
        {"content": "文档1内容...", "score": 0.95},
        {"content": "文档2内容...", "score": 0.90},
        {"content": "文档3内容...", "score": 0.85},
        {"content": "文档4内容...", "score": 0.80},
        {"content": "文档5内容...", "score": 0.75},
        {"content": "文档6内容...", "score": 0.70},
        {"content": "文档7内容...", "score": 0.65},
        {"content": "文档8内容...", "score": 0.60},
    ]

    # 上下文窗口限制（假设最多3个文档）
    max_context_docs = 3
    min_score_threshold = 0.70

    # 策略1：基于Top-K
    context_docs = retrieved_docs[:max_context_docs]
    print("策略1：Top-K选择")
    print("-" * 60)
    for i, doc in enumerate(context_docs):
        print(f"{i + 1}. Score: {doc['score']:.2f}")
    print(f"使用文档数: {len(context_docs)}")

    # 策略2：基于阈值
    context_docs = [doc for doc in retrieved_docs if doc['score'] >= min_score_threshold]
    print("\n策略2：基于阈值过滤")
    print("-" * 60)
    for i, doc in enumerate(context_docs):
        print(f"{i + 1}. Score: {doc['score']:.2f}")
    print(f"使用文档数: {len(context_docs)}")

    # 策略3：综合（Top-K + 阈值）
    context_docs = [
        doc for doc in retrieved_docs[:max_context_docs]
        if doc['score'] >= min_score_threshold
    ]
    print("\n策略3：综合策略")
    print("-" * 60)
    for i, doc in enumerate(context_docs):
        print(f"{i + 1}. Score: {doc['score']:.2f}")
    print(f"使用文档数: {len(context_docs)}")

    # 构建上下文
    def build_context(docs: list, max_tokens: int = 1000) -> str:
        """构建上下文"""
        context_parts = []
        current_tokens = 0

        for doc in docs:
            # 估算tokens（粗略：1个词≈1.3个tokens）
            doc_tokens = len(doc['content']) * 1.3
            if current_tokens + doc_tokens > max_tokens:
                break
            context_parts.append(doc['content'])
            current_tokens += doc_tokens

        return "\n\n".join(context_parts)

    context = build_context(retrieved_docs, max_tokens=200)
    estimated_tokens = len(context) * 1.3

    print("\n上下文构建:")
    print("-" * 60)
    print(f"上下文长度: {len(context)} 字符")
    print(f"估计tokens: {estimated_tokens:.0f}")
    print(f"Token利用率: {estimated_tokens / 200 * 100:.1f}%")

    print("\n✓ 上下文窗口管理测试成功")

    print()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("检索策略测试套件")
    print("=" * 60 + "\n")

    # 测试1：Top-K检索
    test_top_k_retrieval()

    # 测试2：BM25检索
    test_bm25_retrieval()

    # 测试3：混合检索
    test_hybrid_retrieval()

    # 测试4：查询扩展
    test_query_expansion()

    # 测试5：上下文窗口
    test_context_window()

    print("=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n如果所有测试都通过，恭喜你完成了阶段三的学习！")
    print("下一阶段：RAG系统集成")


if __name__ == "__main__":
    main()
