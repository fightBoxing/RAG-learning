#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM工具函数
支持多种大模型API：OpenAI、智谱AI等
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

# 加载.env文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class LLMClient:
    """
    LLM客户端
    支持OpenAI兼容的API（包括智谱AI等）
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        top_p: float = 1.0
    ):
        """
        初始化LLM客户端

        Args:
            api_key: API密钥，默认从环境变量OPENAI_API_KEY读取
            base_url: API基础URL，默认从环境变量OPENAI_BASE_URL读取
            model: 模型名称，默认从环境变量OPENAI_MODEL读取
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: nucleus采样参数
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        if not self.api_key:
            raise ValueError("API密钥未设置，请设置OPENAI_API_KEY环境变量")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False
    ) -> str:
        """
        对话生成

        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: nucleus采样参数
            stream: 是否流式输出

        Returns:
            生成的文本
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            top_p=top_p if top_p is not None else self.top_p,
            stream=stream
        )

        if stream:
            return response  # 返回流式响应
        else:
            return response.choices[0].message.content

    def simple_chat(
        self,
        user_message: str,
        system_message: str = "你是一个乐于助人的AI助手。",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        简单对话

        Args:
            user_message: 用户消息
            system_message: 系统消息
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            生成的文本
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        return self.chat(messages, temperature, max_tokens)

    def rag_generate(
        self,
        context: str,
        question: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        RAG生成

        Args:
            context: 上下文
            question: 问题
            system_message: 系统消息
            max_tokens: 最大token数

        Returns:
            生成的答案
        """
        if system_message is None:
            system_message = "你是一个专业的AI助手。基于提供的上下文回答问题。"

        prompt = f"""基于以下上下文回答问题。

{context}

问题: {question}

回答:"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        return self.chat(messages, max_tokens=max_tokens)

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }


def create_zhipu_client() -> LLMClient:
    """
    创建智谱AI客户端

    Returns:
        LLMClient实例
    """
    return LLMClient(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model="glm-4-flash"
    )


def create_openai_client() -> LLMClient:
    """
    创建OpenAI客户端

    Returns:
        LLMClient实例
    """
    return LLMClient(
        base_url="https://api.openai.com/v1",
        model="gpt-3.5-turbo"
    )


def create_client_from_env() -> LLMClient:
    """
    从环境变量创建客户端
    根据环境变量自动选择模型

    Returns:
        LLMClient实例
    """
    return LLMClient()


# 测试代码
if __name__ == "__main__":
    # 测试LLM客户端
    try:
        print("测试LLM客户端")
        print("=" * 60)

        # 从环境变量创建客户端
        client = create_client_from_env()

        # 显示模型信息
        model_info = client.get_model_info()
        print("\n模型信息:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")

        # 测试简单对话
        print("\n测试简单对话:")
        print("-" * 60)
        question = "什么是RAG技术？"
        print(f"问题: {question}")

        answer = client.simple_chat(question)
        print(f"回答: {answer}")

        # 测试RAG生成
        print("\n测试RAG生成:")
        print("-" * 60)
        context = "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI技术。"
        question = "什么是RAG？"

        answer = client.rag_generate(context, question)
        print(f"问题: {question}")
        print(f"回答: {answer}")

        print("\n✓ LLM客户端测试成功")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        print("\n提示:")
        print("1. 请检查.env文件中的API配置")
        print("2. 确保已安装openai包: pip install openai")
        print("3. 对于智谱AI，确保API格式正确")
