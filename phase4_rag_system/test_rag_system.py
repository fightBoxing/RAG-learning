#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统集成测试 - 验证完整RAG流程
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import time


@dataclass
class Document:
    """文档类"""
    id: str
    content: str
    score: float


class RAGSystem:
    """简单的RAG系统"""

    def __init__(self):
        """初始化RAG系统"""
        self.knowledge_base = [
            {
                "id": "doc1",
                "content": "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI技术。"
            },
            {
                "id": "doc2",
                "content": "向量数据库用于存储和检索文本向量，支持快速相似度搜索。"
            },
            {
                "id": "doc3",
                "content": "Embedding模型将文本转换为数值向量，相似文本在向量空间中距离更近。"
            },
            {
                "id": "doc4",
                "content": "LLM（Large Language Model）大语言模型可以基于上下文生成高质量的文本。"
            },
            {
                "id": "doc5",
                "content": "文档分块是将长文档切分为多个小块，每个块包含一定的token数量。"
            }
        ]
        self.conversation_history = []

    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """检索相关文档（模拟）"""
        # 简单的关键词匹配（实际应使用向量化检索）
        query_keywords = set(query.split())
        scored_docs = []

        for doc in self.knowledge_base:
            doc_keywords = set(doc["content"].split())
            # 计算重叠度作为相似度
            overlap = len(query_keywords & doc_keywords)
            score = overlap / max(len(query_keywords), 1)

            scored_docs.append(Document(
                id=doc["id"],
                content=doc["content"],
                score=score
            ))

        # 排序并返回Top-K
        scored_docs.sort(key=lambda x: x.score, reverse=True)
        return scored_docs[:top_k]

    def build_context(self, documents: List[Document]) -> str:
        """构建上下文"""
        context_parts = []
        for doc in documents:
            context_parts.append(f"[Source {doc.id}]\n{doc.content}")
        return "\n\n".join(context_parts)

    def build_prompt(self, context: str, question: str) -> str:
        """构建Prompt"""
        prompt = f"""基于以下上下文回答问题。如果上下文中没有足够的信息，请明确说明。

{context}

问题: {question}

回答:"""
        return prompt

    def generate(self, prompt: str) -> str:
        """生成答案（模拟）"""
        # 实际应调用LLM API
        # 这里使用简单的规则模拟
        if "RAG" in prompt:
            return "RAG是一种结合检索和生成的AI技术，可以提高大模型回答的准确性。"
        elif "向量" in prompt:
            return "向量数据库用于存储文本向量，支持快速相似度搜索。"
        elif "Embedding" in prompt:
            return "Embedding模型将文本转换为数值向量，用于计算文本相似度。"
        elif "LLM" in prompt:
            return "LLM大语言模型可以基于上下文生成高质量的文本。"
        else:
            return "抱歉，我无法从提供的上下文中找到相关信息。"

    def query(self, question: str) -> Dict:
        """执行完整查询"""
        start_time = time.time()

        # 检索
        retrieved_docs = self.retrieve(question, top_k=3)

        # 构建上下文
        context = self.build_context(retrieved_docs)

        # 构建Prompt
        prompt = self.build_prompt(context, question)

        # 生成答案
        answer = self.generate(prompt)

        # 计算耗时
        elapsed_time = time.time() - start_time

        # 返回结果
        return {
            "question": question,
            "answer": answer,
            "sources": [doc.id for doc in retrieved_docs],
            "retrieved_docs": retrieved_docs,
            "elapsed_time": elapsed_time
        }

    def chat(self, question: str) -> Dict:
        """多轮对话"""
        # 执行查询
        result = self.query(question)

        # 更新对话历史
        self.conversation_history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"]
        })

        # 添加历史到结果
        result["conversation_length"] = len(self.conversation_history)

        return result

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []


def test_basic_rag():
    """测试基础RAG流程"""
    print("=" * 60)
    print("测试1：基础RAG流程")
    print("=" * 60)

    # 创建RAG系统
    rag = RAGSystem()

    print("✓ RAG系统初始化成功")

    # 测试查询
    question = "什么是RAG技术？"
    print(f"\n问题: {question}")

    result = rag.query(question)

    print(f"\n答案: {result['answer']}")
    print(f"来源: {result['sources']}")
    print(f"耗时: {result['elapsed_time']:.3f}秒")

    print("\n✓ 基础RAG流程测试成功")
    print()


def test_prompt_templates():
    """测试Prompt模板"""
    print("=" * 60)
    print("测试2：Prompt模板设计")
    print("=" * 60)

    # 基础Prompt
    basic_prompt = """基于以下上下文回答问题。

{context}

问题: {question}

回答:"""

    print("基础Prompt:")
    print("-" * 60)
    print(basic_prompt)

    # 优化Prompt
    optimized_prompt = """你是一个专业的AI助手。基于提供的上下文，准确回答用户的问题。
如果上下文中没有足够的信息，请明确说明"根据提供的上下文，我无法回答这个问题"。
提供简洁、准确的答案，并在适当的地方标注信息来源。

{context}

问题: {question}

回答:"""

    print("\n优化Prompt:")
    print("-" * 60)
    print(optimized_prompt)

    # Few-shot Prompt
    few_shot_prompt = """基于以下上下文回答问题。如果上下文信息不足，请明确说明。

示例:
上下文: RAG是检索增强生成技术。
问题: 什么是RAG？
回答: RAG是检索增强生成技术。

{context}

问题: {question}

回答:"""

    print("\nFew-shot Prompt:")
    print("-" * 60)
    print(few_shot_prompt)

    print("\n✓ Prompt模板测试成功")
    print()


def test_context_building():
    """测试上下文构建"""
    print("=" * 60)
    print("测试3：上下文构建")
    print("=" * 60)

    # 示例文档
    documents = [
        Document(id="doc1", content="RAG是检索增强生成技术。", score=0.95),
        Document(id="doc2", content="向量数据库用于存储文本向量。", score=0.90),
        Document(id="doc3", content="Embedding模型将文本转换为向量。", score=0.85),
    ]

    # 策略1：简单拼接
    context1 = "\n\n".join([doc.content for doc in documents])
    print("策略1：简单拼接")
    print("-" * 60)
    print(context1)

    # 策略2：带ID标注
    context2 = "\n\n".join([f"[Source {doc.id}]\n{doc.content}" for doc in documents])
    print("\n策略2：带ID标注")
    print("-" * 60)
    print(context2)

    # 策略3：带相似度
    context3 = "\n\n".join([f"[Source {doc.id}, Score: {doc.score:.2f}]\n{doc.content}" for doc in documents])
    print("\n策略3：带相似度")
    print("-" * 60)
    print(context3)

    print("\n✓ 上下文构建测试成功")
    print()


def test_multi_turn_conversation():
    """测试多轮对话"""
    print("=" * 60)
    print("测试4：多轮对话")
    print("=" * 60)

    # 创建RAG系统
    rag = RAGSystem()

    # 第一轮对话
    print("第一轮对话:")
    print("-" * 60)
    q1 = "什么是RAG技术？"
    print(f"用户: {q1}")
    r1 = rag.chat(q1)
    print(f"助手: {r1['answer']}")
    print(f"对话轮数: {r1['conversation_length']}")

    # 第二轮对话
    print("\n第二轮对话:")
    print("-" * 60)
    q2 = "向量数据库有什么用？"
    print(f"用户: {q2}")
    r2 = rag.chat(q2)
    print(f"助手: {r2['answer']}")
    print(f"对话轮数: {r2['conversation_length']}")

    # 第三轮对话
    print("\n第三轮对话:")
    print("-" * 60)
    q3 = "什么是Embedding？"
    print(f"用户: {q3}")
    r3 = rag.chat(q3)
    print(f"助手: {r3['answer']}")
    print(f"对话轮数: {r3['conversation_length']}")

    # 显示对话历史
    print("\n对话历史:")
    print("-" * 60)
    for i, turn in enumerate(rag.conversation_history, 1):
        print(f"第{i}轮:")
        print(f"  问题: {turn['question']}")
        print(f"  答案: {turn['answer']}")
        print(f"  来源: {turn['sources']}")

    print("\n✓ 多轮对话测试成功")
    print()


def test_answer_tracing():
    """测试答案溯源"""
    print("=" * 60)
    print("测试5：答案溯源")
    print("=" * 60)

    # 创建RAG系统
    rag = RAGSystem()

    # 查询
    question = "什么是RAG技术？"
    result = rag.query(question)

    print(f"问题: {question}")
    print(f"\n答案: {result['answer']}")
    print(f"\n来源文档: {result['sources']}")

    # 显示检索到的文档详情
    print("\n检索到的文档:")
    print("-" * 60)
    for doc in result['retrieved_docs']:
        print(f"ID: {doc.id}")
        print(f"内容: {doc.content}")
        print(f"相似度: {doc.score:.4f}")
        print()

    print("✓ 答案溯源测试成功")
    print()


def test_error_handling():
    """测试错误处理"""
    print("=" * 60)
    print("测试6：错误处理")
    print("=" * 60)

    # 创建RAG系统
    rag = RAGSystem()

    # 测试1：检索失败（没有相关文档）
    print("测试1：检索无相关文档")
    print("-" * 60)
    q1 = "什么是量子力学？"  # 知识库中没有相关内容
    r1 = rag.query(q1)
    print(f"问题: {q1}")
    print(f"答案: {r1['answer']}")
    print(f"来源: {r1['sources']}")

    # 测试2：空问题
    print("\n测试2：空问题")
    print("-" * 60)
    q2 = ""
    if not q2.strip():
        print("检测到空问题，返回提示信息")
        print("请提供有效的问题。")
    else:
        r2 = rag.query(q2)
        print(f"答案: {r2['answer']}")

    # 测试3：超长问题
    print("\n测试3：超长问题")
    print("-" * 60)
    q3 = "什么是" + "很长的" * 50 + "问题？"
    if len(q3) > 1000:
        print("检测到超长问题，返回提示信息")
        print("问题过长，请精简后重新提问。")
    else:
        r3 = rag.query(q3)
        print(f"答案: {r3['answer']}")

    print("\n✓ 错误处理测试成功")
    print()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("RAG系统集成测试套件")
    print("=" * 60 + "\n")

    # 测试1：基础RAG流程
    test_basic_rag()

    # 测试2：Prompt模板
    test_prompt_templates()

    # 测试3：上下文构建
    test_context_building()

    # 测试4：多轮对话
    test_multi_turn_conversation()

    # 测试5：答案溯源
    test_answer_tracing()

    # 测试6：错误处理
    test_error_handling()

    print("=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n如果所有测试都通过，恭喜你完成了阶段四的学习！")
    print("下一阶段：评估与优化")


if __name__ == "__main__":
    main()
