#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智谱AI测试代码
验证智谱大模型API是否正常工作
"""

import os
import sys
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.llm_utils import create_client_from_env


def test_zhipu_connection():
    """测试智谱AI连接"""
    print("=" * 60)
    print("测试1：智谱AI连接")
    print("=" * 60)

    try:
        # 检查环境变量
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("OPENAI_MODEL")

        print(f"API密钥: {'*' * (len(api_key) - 8)}{api_key[-8:]}")
        print(f"API地址: {base_url}")
        print(f"模型: {model}")

        if not api_key:
            print("\n✗ 未设置OPENAI_API_KEY环境变量")
            return False

        print("\n✓ 环境变量配置正确")
        return True

    except Exception as e:
        print(f"✗ 检查失败: {e}")
        return False


def test_simple_chat():
    """测试简单对话"""
    print("\n" + "=" * 60)
    print("测试2：简单对话")
    print("=" * 60)

    try:
        # 创建客户端
        client = create_client_from_env()

        # 测试对话
        question = "你好！请用一句话介绍一下RAG技术。"
        print(f"\n问题: {question}")

        answer = client.simple_chat(question, max_tokens=100)
        print(f"\n回答: {answer}")

        print("\n✓ 简单对话测试成功")
        return True

    except Exception as e:
        print(f"✗ 对话失败: {e}")
        print("\n可能的原因:")
        print("1. API密钥无效")
        print("2. API地址配置错误")
        print("3. 模型名称不正确")
        print("4. 网络连接问题")
        return False


def test_rag_generation():
    """测试RAG生成"""
    print("\n" + "=" * 60)
    print("测试3：RAG生成")
    print("=" * 60)

    try:
        # 创建客户端
        client = create_client_from_env()

        # 测试RAG生成
        context = """
        RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI技术。
        它的工作流程包括：文档加载、向量化、检索、生成四个步骤。
        RAG可以减少大模型的幻觉问题，提高答案的准确性。
        """

        question = "RAG技术的优势是什么？"
        print(f"\n上下文: {context[:100]}...")
        print(f"\n问题: {question}")

        answer = client.rag_generate(context, question, max_tokens=200)
        print(f"\n回答: {answer}")

        print("\n✓ RAG生成测试成功")
        return True

    except Exception as e:
        print(f"✗ RAG生成失败: {e}")
        return False


def test_multi_turn_conversation():
    """测试多轮对话"""
    print("\n" + "=" * 60)
    print("测试4：多轮对话")
    print("=" * 60)

    try:
        # 创建客户端
        client = create_client_from_env()

        # 第一轮
        print("\n第一轮对话:")
        print("-" * 60)
        q1 = "什么是向量数据库？"
        print(f"用户: {q1}")
        a1 = client.simple_chat(q1)
        print(f"助手: {a1}")

        # 第二轮
        print("\n第二轮对话:")
        print("-" * 60)
        q2 = "它有什么优势？"
        print(f"用户: {q2}")
        a2 = client.simple_chat(q2)
        print(f"助手: {a2}")

        print("\n✓ 多轮对话测试成功")
        return True

    except Exception as e:
        print(f"✗ 多轮对话失败: {e}")
        return False


def test_temperature_control():
    """测试温度参数控制"""
    print("\n" + "=" * 60)
    print("测试5：温度参数控制")
    print("=" * 60)

    try:
        # 创建客户端
        client = create_client_from_env()

        question = "请简述RAG技术。"

        # 测试不同温度
        temperatures = [0.0, 0.5, 1.0]

        for temp in temperatures:
            print(f"\n温度: {temp}")
            print("-" * 60)

            answer = client.simple_chat(
                question,
                temperature=temp,
                max_tokens=100
            )
            print(f"回答: {answer}")

        print("\n✓ 温度控制测试成功")
        return True

    except Exception as e:
        print(f"✗ 温度控制失败: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("智谱AI测试套件")
    print("=" * 60)

    # 运行所有测试
    tests = [
        ("环境配置检查", test_zhipu_connection),
        ("简单对话", test_simple_chat),
        ("RAG生成", test_rag_generation),
        ("多轮对话", test_multi_turn_conversation),
        ("温度控制", test_temperature_control),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n【{test_name}】")
        success = test_func()
        results.append((test_name, success))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\n通过: {passed}/{total}")

    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {test_name}: {status}")

    if passed == total:
        print("\n🎉 所有测试通过！智谱AI配置正确。")
    else:
        print("\n⚠ 部分测试失败，请检查配置。")
        print("\n故障排查建议:")
        print("1. 检查.env文件中的OPENAI_API_KEY是否正确")
        print("2. 检查OPENAI_BASE_URL是否为: https://open.bigmodel.cn/api/paas/v4/")
        print("3. 检查OPENAI_MODEL是否为: glm-4-flash 或其他智谱模型")
        print("4. 确保网络连接正常，可以访问智谱AI API")
        print("5. 查看智谱AI官网: https://open.bigmodel.cn/")


if __name__ == "__main__":
    main()
