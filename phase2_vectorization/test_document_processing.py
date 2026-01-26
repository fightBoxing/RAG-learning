#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档处理测试 - 验证文档解析和分块功能

"""

import os
from typing import List, Dict, Any
import re


def clean_pdf_text(text: str) -> str:
    """清洗从PDF提取的文本"""
    # 去除多余的空白字符
    text = re.sub(r'\s+', ' ', text)

    # 去除特殊字符（保留中文、英文、数字、常用标点）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。？！、；：""''（）《》\s]', '', text)

    # 去除页眉页脚常见标记
    text = re.sub(r'第\s*\d+\s*页', '', text)
    text = re.sub(r'Page\s*\d+', '', text)

    # 去除首尾空白
    text = text.strip()

    return text


def test_pdf_cleaning(pdf_path: str = None):
    """测试PDF文件清洗功能"""
    print("=" * 60)
    print("测试：PDF文件清洗")
    print("=" * 60)

    try:
        import fitz  # PyMuPDF

        # 如果没有提供PDF路径，创建一个测试PDF
        if pdf_path is None:
            # 创建一个测试PDF文件
            pdf_path = "test_document.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text(
                (50, 72),
                "这是一个测试PDF文档！！！\n\n包含  多余  的  空格...\n\n"
                "RAG（Retrieval-Augmented Generation）是一种AI技术。\n"
                "第1页\n\n"
                "它可以减少大模型的幻觉问题。\n"
                "第2页\n\n"
                "Page 3\n\n"
                "还有一些特殊字符：@#$%^&*()\n"
                "End of test document."
            )
            doc.save(pdf_path)
            doc.close()
            print(f"已创建测试PDF文件: {pdf_path}")

        # 打开PDF文件
        pdf_document = fitz.open(pdf_path)

        print(f"\nPDF文件信息:")
        print(f"- 文件路径: {pdf_path}")
        print(f"- 页数: {pdf_document.page_count}")
        print(f"- 元数据: {pdf_document.metadata}")

        # 提取所有页面的文本
        all_text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            page_text = page.get_text()
            all_text += f"\n--- 第{page_num + 1}页原始文本 ---\n{page_text}"

        pdf_document.close()

        print("\n原始提取文本:")
        print("-" * 60)
        print(all_text)
        print()

        # 清洗文本
        cleaned_text = clean_pdf_text(all_text)

        print("清洗后文本:")
        print("-" * 60)
        print(cleaned_text)
        print()

        # 统计
        print("清洗效果:")
        print(f"- 原始长度: {len(all_text)}")
        print(f"- 清洗后长度: {len(cleaned_text)}")
        print(f"- 减少字符: {len(all_text) - len(cleaned_text)}")
        print(f"- 压缩率: {(1 - len(cleaned_text) / len(all_text)) * 100:.2f}%")

        # 清理测试文件
        if pdf_path.startswith("/tmp/"):
            os.remove(pdf_path)
            print(f"\n已删除测试文件: {pdf_path}")

        print("\n✓ PDF文件清洗测试成功")

    except ImportError:
        print("✗ 未安装PyMuPDF库，请运行: pip install pymupdf")
    except Exception as e:
        print(f"✗ 测试失败: {e}")

    print()


def test_text_cleaning():
    """测试文本清洗功能"""
    print("=" * 60)
    print("测试1：文本清洗")
    print("=" * 60)

    # 示例文本
    dirty_text = """
    这是一个测试文本！！！
    包含  多余  的  空格...
    还有一些特殊字符：@#$%^&*()
    <html>HTML标签</html>
    英文：Hello World 123
    """

    print("原始文本:")
    print(dirty_text)
    print()

    # 清洗函数
    def clean_text(text: str) -> str:
        """清洗文本"""
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)

        # 去除特殊字符（保留中文、英文、数字、常用标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。？！、；：“”''（）《》\s]', '', text)

        # 统一空白字符
        text = re.sub(r'\s+', ' ', text)

        # 去除首尾空白
        text = text.strip()

        return text

    cleaned_text = clean_text(dirty_text)

    print("清洗后文本:")
    print(cleaned_text)
    print()

    # 统计
    print("清洗效果:")
    print(f"- 原始长度: {len(dirty_text)}")
    print(f"- 清洗后长度: {len(cleaned_text)}")
    print(f"- 减少字符: {len(dirty_text) - len(cleaned_text)}")

    print("\n✓ 文本清洗测试成功")
    print()


def test_chunking_strategies():
    """测试分块策略"""
    print("=" * 60)
    print("测试2：文档分块策略")
    print("=" * 60)

    # 示例文档
    sample_text = """
    RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI技术。
    它可以减少大模型的幻觉问题，提高答案的准确性。
    RAG的工作流程包括文档处理、向量化、检索、生成四个步骤。
    文档处理包括文档加载、文本清洗、文档分块等环节。
    向量化是将文本转换为数值向量的过程。
    检索是根据查询找到相关文档的过程。
    生成是使用检索到的上下文生成答案的过程。
    RAG技术广泛应用于问答系统、知识库、文档分析等领域。
    """

    print("示例文档:")
    print(sample_text)
    print()

    # 策略1：固定大小分块
    def fixed_size_chunk(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
        """固定大小分块"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += (chunk_size - overlap)
        return chunks

    chunks_fixed = fixed_size_chunk(sample_text.strip(), chunk_size=50, overlap=10)
    print("策略1：固定大小分块 (chunk_size=50, overlap=10)")
    print("-" * 60)
    for i, chunk in enumerate(chunks_fixed):
        print(f"Chunk {i + 1}: {chunk}")
    print(f"共 {len(chunks_fixed)} 个Chunk")
    print()

    # 策略2：基于句子分块
    def sentence_chunk(text: str) -> List[str]:
        """基于句子分块"""
        # 使用正则表达式分割句子
        sentences = re.split(r'[。！？\n]', text)
        # 过滤空句子
        chunks = [s.strip() for s in sentences if s.strip()]
        return chunks

    chunks_sentence = sentence_chunk(sample_text.strip())
    print("策略2：基于句子分块")
    print("-" * 60)
    for i, chunk in enumerate(chunks_sentence):
        print(f"Chunk {i + 1}: {chunk}")
    print(f"共 {len(chunks_sentence)} 个Chunk")
    print()

    # 策略3：基于段落分块
    def paragraph_chunk(text: str) -> List[str]:
        """基于段落分块"""
        paragraphs = text.strip().split('\n\n')
        chunks = [p.strip() for p in paragraphs if p.strip()]
        return chunks

    chunks_paragraph = paragraph_chunk(sample_text.strip())
    print("策略3：基于段落分块")
    print("-" * 60)
    for i, chunk in enumerate(chunks_paragraph):
        print(f"Chunk {i + 1}: {chunk}")
    print(f"共 {len(chunks_paragraph)} 个Chunk")
    print()

    print("✓ 分块策略测试成功")
    print()


def test_embedding_comparison():
    """测试不同Embedding模型对比"""
    print("=" * 60)
    print("测试3：Embedding模型对比")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        # 测试文本对
        text_pairs = [
            ("RAG是一种AI技术", "检索增强生成是人工智能技术"),
            ("RAG是一种AI技术", "今天天气很好"),
            ("Python是一种编程语言", "编程语言有很多种"),
        ]

        # 测试模型列表
        models_to_test = [
            {
                "name": "all-MiniLM-L6-v2",
                "model": "all-MiniLM-L6-v2",
                "dims": 384
            },
            {
                "name": "bge-large-zh",
                "model": "BAAI/bge-large-zh",
                "dims": 1024
            }

        ]

        for model_info in models_to_test:
            print(f"\n模型: {model_info['name']}")
            print(f"维度: {model_info['dims']}")
            print("-" * 60)

            model = SentenceTransformer(model_info['model'])

            for text1, text2 in text_pairs:
                embeddings = model.encode([text1, text2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

                print(f"\n文本1: {text1}")
                print(f"文本2: {text2}")
                print(f"相似度: {similarity:.4f}")

        print("\n✓ Embedding模型对比测试成功")

    except Exception as e:
        print(f"✗ 测试失败: {e}")

    print()


def test_vector_database():
    """测试向量数据库"""
    print("=" * 60)
    print("测试4：向量数据库（Chroma）")
    print("=" * 60)

    try:
        import chromadb

        # 创建客户端
        client = chromadb.Client()

        # client = chromadb.PersistentClient(path="chromadb-v1")

        # 创建集合
        collection = client.create_collection(
            name="test_collection",
            metadata={"hnsw:space": "cosine"}
        )

        print("✓ 创建ChromaDB集合成功")

        # 测试文档
        documents = [
            "RAG是检索增强生成技术",
            "向量数据库用于存储文本向量",
            "Embedding模型将文本转换为向量"
        ]

        # 添加文档
        collection.add(
            documents=documents,
            metadatas=[{"source": "test1"}, {"source": "test2"}, {"source": "test3"}],
            ids=["doc1", "doc2", "doc3"]
        )

        print("✓ 添加文档到向量数据库成功")

        # 查询文档数量
        count = collection.count()
        print(f"文档数量: {count}")

        # 测试查询
        query_text = "什么是RAG？"
        print(f"\n查询: {query_text}")

        # 注意：实际查询需要向量化，这里只测试数据库操作
        results = collection.get(
            ids=["doc1", "doc2"],
            include=["documents", "metadatas"]
        )

        print("\n查询结果:")
        print("-" * 60)
        for i, doc in enumerate(results['documents']):
            print(f"文档 {i + 1}: {doc}")
            print(f"元数据: {results['metadatas'][i]}")

        print("\n✓ 向量数据库测试成功")

    except Exception as e:
        print(f"✗ 测试失败: {e}")

    print()


def test_batch_processing():
    """测试批量处理"""
    print("=" * 60)
    print("测试5：批量向量化")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer
        import time
        from tqdm import tqdm

        # 加载模型
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # 生成测试数据
        test_documents = [
            "这是第1个测试文档，用于测试批量向量化功能。" * i
            for i in range(1, 101)
        ]

        print(f"文档总数: {len(test_documents)}")

        # 测试批量向量化
        start_time = time.time()
        embeddings = model.encode(test_documents, show_progress_bar=True)
        end_time = time.time()

        batch_time = end_time - start_time
        throughput = len(test_documents) / batch_time

        print("\n批量处理结果:")
        print("-" * 60)
        print(f"总时间: {batch_time:.2f}秒")
        print(f"吞吐量: {throughput:.2f} 文档/秒")
        print(f"平均每个文档: {batch_time / len(test_documents) * 1000:.2f}毫秒")
        print(f"向量形状: {embeddings.shape}")

        # 性能评估
        if throughput > 100:
            print(f"\n✓ 性能良好 ({throughput:.2f} 文档/秒)")
        elif throughput > 50:
            print(f"\n✓ 性能可接受 ({throughput:.2f} 文档/秒)")
        else:
            print(f"\n⚠ 性能需要优化 ({throughput:.2f} 文档/秒)")

        print("\n✓ 批量处理测试成功")

    except Exception as e:
        print(f"✗ 测试失败: {e}")

    print()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("文档处理与向量化测试套件")
    print("=" * 60 + "\n")

    # 测试：PDF文件清洗
    test_pdf_cleaning("1908.10084v1.pdf")

    # 测试1：文本清洗
    test_text_cleaning()

    # 测试2：分块策略
    test_chunking_strategies()

    # 测试3：Embedding模型对比
    test_embedding_comparison()

    # 测试4：向量数据库
    test_vector_database()

    # 测试5：批量处理
    test_batch_processing()

    print("=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n如果所有测试都通过，恭喜你完成了阶段二的学习！")
    print("下一阶段：检索策略优化")


if __name__ == "__main__":
    main()
