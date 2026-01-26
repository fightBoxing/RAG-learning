#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档分块策略测试 - 使用LangChain实现

本脚本演示如何使用LangChain的文本分割器来实现多种分块策略。
"""

from typing import List


def test_chunking_strategies():
    """测试分块策略（使用LangChain实现）"""
    print("=" * 60)
    print("文档分块策略测试（LangChain实现）")
    print("=" * 60)

    try:
        from langchain_text_splitters import (
            CharacterTextSplitter,
            RecursiveCharacterTextSplitter,
            TokenTextSplitter,
        )

        # 示例文档
        sample_text = """RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI技术。
它可以减少大模型的幻觉问题，提高答案的准确性。
RAG的工作流程包括文档处理、向量化、检索、生成四个步骤。
文档处理包括文档加载、文本清洗、文档分块等环节。
向量化是将文本转换为数值向量的过程。
检索是根据查询找到相关文档的过程。
生成是使用检索到的上下文生成答案的过程。
RAG技术广泛应用于问答系统、知识库、文档分析等领域。"""

        print("示例文档:")
        print(sample_text)
        print()

        # 策略1：CharacterTextSplitter - 基于字符的固定大小分块
        print("策略1：CharacterTextSplitter（基于字符的固定大小分块）")
        print("-" * 60)
        char_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=50,
            chunk_overlap=10,
            length_function=len
        )
        chunks_char = char_splitter.split_text(sample_text)
        for i, chunk in enumerate(chunks_char):
            print(f"Chunk {i + 1}: {chunk}")
        print(f"共 {len(chunks_char)} 个Chunk")
        print()

        # 策略2：RecursiveCharacterTextSplitter - 递归字符分块（推荐）
        print("策略2：RecursiveCharacterTextSplitter（递归字符分块，推荐使用）")
        print("-" * 60)
        recursive_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
            chunk_size=80,
            chunk_overlap=20,
            length_function=len
        )
        chunks_recursive = recursive_splitter.split_text(sample_text)
        for i, chunk in enumerate(chunks_recursive):
            print(f"Chunk {i + 1}: {chunk}")
        print(f"共 {len(chunks_recursive)} 个Chunk")
        print()

        # 策略3：TokenTextSplitter - 基于Token的分块
        print("策略3：TokenTextSplitter（基于Token的分块）")
        print("-" * 60)
        try:
            token_splitter = TokenTextSplitter(
                chunk_size=30,
                chunk_overlap=5
            )
            chunks_token = token_splitter.split_text(sample_text)
            for i, chunk in enumerate(chunks_token):
                print(f"Chunk {i + 1}: {chunk}")
            print(f"共 {len(chunks_token)} 个Chunk")
        except Exception as e:
            print(f"TokenTextSplitter需要tiktoken库: {e}")
        print()

        # 策略4：自定义中文句子分块
        print("策略4：基于中文句子分块（自定义分隔符）")
        print("-" * 60)
        chinese_splitter = RecursiveCharacterTextSplitter(
            separators=["。", "！", "？", "\n"],
            chunk_size=100,
            chunk_overlap=0,
            keep_separator=True,
            length_function=len
        )
        chunks_chinese = chinese_splitter.split_text(sample_text)
        for i, chunk in enumerate(chunks_chinese):
            print(f"Chunk {i + 1}: {chunk}")
        print(f"共 {len(chunks_chinese)} 个Chunk")
        print()

        # 分块策略对比总结
        print("=" * 60)
        print("分块策略对比总结：")
        print("-" * 60)
        print(f"CharacterTextSplitter:         {len(chunks_char)} 个Chunk")
        print(f"RecursiveCharacterTextSplitter: {len(chunks_recursive)} 个Chunk")
        print(f"中文句子分块:                   {len(chunks_chinese)} 个Chunk")
        print()
        print("推荐使用 RecursiveCharacterTextSplitter，因为它：")
        print("  1. 支持多级分隔符，优先保持语义完整性")
        print("  2. 对中英文混合文本处理效果好")
        print("  3. 可自定义分隔符顺序")

        print("\n✓ 分块策略测试成功（LangChain）")

    except ImportError as e:
        print(f"✗ 未安装LangChain库，请运行: pip install langchain-text-splitters")
        print(f"  错误详情: {e}")
    except Exception as e:
        print(f"✗ 测试失败: {e}")

    print()


def test_advanced_splitters():
    """测试LangChain高级分割器"""
    print("=" * 60)
    print("高级分割器测试（LangChain）")
    print("=" * 60)

    try:
        from langchain_text_splitters import (
            RecursiveCharacterTextSplitter,
            MarkdownHeaderTextSplitter,
            PythonCodeTextSplitter,
            Language,
        )

        # 测试Markdown分割
        print("\n1. Markdown文本分割")
        print("-" * 60)
        markdown_text = """# 标题一

这是第一段内容。

## 标题二

这是第二段内容，包含一些重要信息。

### 标题三

- 列表项1
- 列表项2
- 列表项3

## 另一个标题

最后一段内容。
"""
        md_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=100,
            chunk_overlap=20
        )
        md_chunks = md_splitter.split_text(markdown_text)
        for i, chunk in enumerate(md_chunks):
            print(f"Chunk {i + 1}:\n{chunk}\n")
        print(f"共 {len(md_chunks)} 个Chunk")

        # 测试Python代码分割
        print("\n2. Python代码分割")
        print("-" * 60)
        python_code = '''
def hello_world():
    """打招呼函数"""
    print("Hello, World!")

class MyClass:
    """示例类"""
    
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        """打招呼方法"""
        return f"Hello, {self.name}!"

def main():
    """主函数"""
    hello_world()
    obj = MyClass("RAG")
    print(obj.greet())
'''
        py_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=150,
            chunk_overlap=30
        )
        py_chunks = py_splitter.split_text(python_code)
        for i, chunk in enumerate(py_chunks):
            print(f"Chunk {i + 1}:\n{chunk}\n")
        print(f"共 {len(py_chunks)} 个Chunk")

        print("\n✓ 高级分割器测试成功")

    except ImportError as e:
        print(f"✗ 未安装相关库: {e}")
    except Exception as e:
        print(f"✗ 测试失败: {e}")

    print()


def test_document_splitter():
    """测试文档级别的分割"""
    print("=" * 60)
    print("文档级别分割测试（LangChain）")
    print("=" * 60)

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        # 创建示例文档
        documents = [
            Document(
                page_content="RAG是检索增强生成技术，它结合了检索和生成两种方法。",
                metadata={"source": "doc1.txt", "page": 1}
            ),
            Document(
                page_content="向量数据库用于存储和检索文本向量，常见的有Chroma、FAISS、Milvus等。",
                metadata={"source": "doc2.txt", "page": 1}
            ),
            Document(
                page_content="Embedding模型将文本转换为稠密向量，用于语义相似度计算。常用的模型包括OpenAI的text-embedding-ada-002和开源的sentence-transformers系列。",
                metadata={"source": "doc3.txt", "page": 1}
            ),
        ]

        print("原始文档:")
        for i, doc in enumerate(documents):
            print(f"文档 {i + 1}: {doc.page_content[:50]}...")
            print(f"  元数据: {doc.metadata}")
        print()

        # 分割文档
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=10
        )

        split_docs = splitter.split_documents(documents)

        print("分割后的文档:")
        print("-" * 60)
        for i, doc in enumerate(split_docs):
            print(f"Chunk {i + 1}: {doc.page_content}")
            print(f"  元数据: {doc.metadata}")
        print()
        print(f"共 {len(split_docs)} 个Chunk（从 {len(documents)} 个文档分割）")

        print("\n✓ 文档级别分割测试成功")

    except ImportError as e:
        print(f"✗ 未安装相关库: {e}")
    except Exception as e:
        print(f"✗ 测试失败: {e}")

    print()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("LangChain 文档分块策略测试套件")
    print("=" * 60 + "\n")

    # 测试1：基础分块策略
    test_chunking_strategies()

    # 测试2：高级分割器
    test_advanced_splitters()

    # 测试3：文档级别分割
    test_document_splitter()

    print("=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n使用说明:")
    print("1. 安装依赖: pip install langchain-text-splitters langchain-core tiktoken")
    print("2. 运行脚本: python test_chunking_langchain.py")


if __name__ == "__main__":
    main()
