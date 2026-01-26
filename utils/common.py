#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数集合
提供RAG开发过程中常用的工具函数
"""

import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path


def load_env_vars() -> Dict[str, str]:
    """
    加载环境变量

    Returns:
        环境变量字典
    """
    env_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "CHROMA_PERSIST_DIR": os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
    }

    return env_vars


def measure_time(func):
    """
    测量函数执行时间的装饰器

    Args:
        func: 要测量的函数

    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} 执行时间: {elapsed_time:.3f}秒")
        return result
    return wrapper


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separator: str = "\n\n"
) -> List[str]:
    """
    将文本分割成块

    Args:
        text: 要分割的文本
        chunk_size: 每块的大小
        overlap: 块之间的重叠
        separator: 分隔符

    Returns:
        文本块列表
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += (chunk_size - overlap)

    return chunks


def normalize_scores(scores: List[float]) -> List[float]:
    """
    归一化分数到[0, 1]区间

    Args:
        scores: 分数列表

    Returns:
        归一化后的分数列表
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [0.5] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


def calculate_precision_at_k(
    retrieved_docs: List[str],
    relevant_docs: List[str],
    k: int
) -> float:
    """
    计算Precision@K

    Args:
        retrieved_docs: 检索到的文档列表
        relevant_docs: 相关文档列表
        k: Top-K的K值

    Returns:
        Precision@K分数
    """
    top_k = retrieved_docs[:k]
    relevant_in_top_k = set(top_k) & set(relevant_docs)

    return len(relevant_in_top_k) / k if k > 0 else 0.0


def calculate_recall_at_k(
    retrieved_docs: List[str],
    relevant_docs: List[str],
    k: int
) -> float:
    """
    计算Recall@K

    Args:
        retrieved_docs: 检索到的文档列表
        relevant_docs: 相关文档列表
        k: Top-K的K值

    Returns:
        Recall@K分数
    """
    top_k = retrieved_docs[:k]
    relevant_in_top_k = set(top_k) & set(relevant_docs)

    return len(relevant_in_top_k) / len(relevant_docs) if relevant_docs else 0.0


def calculate_mrr(
    retrieved_docs: List[str],
    relevant_docs: List[str]
) -> float:
    """
    计算MRR（Mean Reciprocal Rank）

    Args:
        retrieved_docs: 检索到的文档列表
        relevant_docs: 相关文档列表

    Returns:
        MRR分数
    """
    for i, doc in enumerate(retrieved_docs, 1):
        if doc in relevant_docs:
            return 1.0 / i

    return 0.0


def save_to_file(content: str, filepath: str) -> bool:
    """
    保存内容到文件

    Args:
        content: 要保存的内容
        filepath: 文件路径

    Returns:
        是否保存成功
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"保存文件失败: {e}")
        return False


def load_from_file(filepath: str) -> Optional[str]:
    """
    从文件加载内容

    Args:
        filepath: 文件路径

    Returns:
        文件内容，失败返回None
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"加载文件失败: {e}")
        return None


def format_documents(
    documents: List[Dict[str, Any]],
    include_metadata: bool = False
) -> str:
    """
    格式化文档为字符串

    Args:
        documents: 文档列表
        include_metadata: 是否包含元数据

    Returns:
        格式化后的字符串
    """
    formatted_parts = []

    for i, doc in enumerate(documents, 1):
        content = doc.get("content", "")

        if include_metadata:
            metadata = doc.get("metadata", {})
            metadata_str = ", ".join([f"{k}={v}" for k, v in metadata.items()])
            formatted_parts.append(f"[Doc {i}] {content} ({metadata_str})")
        else:
            formatted_parts.append(f"[Doc {i}] {content}")

    return "\n\n".join(formatted_parts)


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    截断文本到指定长度

    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 截断后添加的后缀

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def extract_entities(text: str, entity_types: Optional[List[str]] = None) -> List[str]:
    """
    简单的实体提取（占位函数）

    Args:
        text: 输入文本
        entity_types: 实体类型列表

    Returns:
        提取的实体列表
    """
    # 这是一个占位函数，实际应使用NER模型
    # 例如：spaCy, HuggingFace Transformers等
    return []


def clean_text(text: str) -> str:
    """
    清洗文本

    Args:
        text: 原始文本

    Returns:
        清洗后的文本
    """
    import re

    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)

    # 去除多余空白
    text = re.sub(r'\s+', ' ', text)

    # 去除首尾空白
    text = text.strip()

    return text


def estimate_tokens(text: str, ratio: float = 1.3) -> int:
    """
    估算文本的token数量

    Args:
        text: 输入文本
        ratio: token/字符比率（默认1.3）

    Returns:
        估算的token数量
    """
    return int(len(text) * ratio)


def print_progress(current: int, total: int, prefix: str = "Progress"):
    """
    打印进度条

    Args:
        current: 当前进度
        total: 总数
        prefix: 前缀文本
    """
    percent = (current / total) * 100
    bar_length = 50
    filled = int(bar_length * current / total)

    bar = '█' * filled + '-' * (bar_length - filled)
    print(f'\r{prefix}: [{bar}] {percent:.1f}% ({current}/{total})', end='')

    if current == total:
        print()


# 测试代码
if __name__ == "__main__":
    # 测试分块
    text = "这是一个很长的文本，用于测试分块功能。" * 50
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    print(f"分块测试: {len(chunks)} 个块")

    # 测试归一化
    scores = [0.1, 0.5, 0.9, 0.3]
    normalized = normalize_scores(scores)
    print(f"归一化测试: {normalized}")

    # 测试评估指标
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc2", "doc4", "doc6"]

    precision = calculate_precision_at_k(retrieved, relevant, k=5)
    recall = calculate_recall_at_k(retrieved, relevant, k=5)
    mrr = calculate_mrr(retrieved, relevant)

    print(f"Precision@5: {precision:.4f}")
    print(f"Recall@5: {recall:.4f}")
    print(f"MRR: {mrr:.4f}")

    print("\n工具函数测试完成")
