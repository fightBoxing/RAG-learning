#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分块策略效果对比测试

本脚本测试不同分块配置下的 RAG 检索效果：
1. 使用 BGE-large-zh 模型进行向量化
2. 将不同配置的分块写入 ChromaDB
3. 通过相似度搜索对比不同配置的检索效果

依赖安装：
pip install langchain-text-splitters chromadb sentence-transformers
"""

import os
import shutil
import time
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass


# ============================================================
# 分块配置定义
# ============================================================
@dataclass
class ChunkConfig:
    """分块配置类"""
    name: str
    chunk_size: int
    chunk_overlap: int
    description: str


# 定义不同的分块配置
CHUNK_CONFIGS = [
    ChunkConfig(
        name="config_small",
        chunk_size=120,
        chunk_overlap=0,
        description="小块配置(120/0) - 句子级别，无重叠"
    ),
    ChunkConfig(
        name="config_qa",
        chunk_size=150,
        chunk_overlap=20,
        description="问答配置(150/20) - 高精度问答场景"
    ),
    ChunkConfig(
        name="config_standard",
        chunk_size=180,
        chunk_overlap=30,
        description="标准配置(180/30) - BGE推荐配置"
    ),
    ChunkConfig(
        name="config_large",
        chunk_size=250,
        chunk_overlap=50,
        description="大块配置(250/50) - 更多上下文"
    ),
]

# 中文分隔符（优先级从高到低）
CHINESE_SEPARATORS = [
    "\n\n",    # 段落
    "\n",      # 换行
    "。",      # 句号
    "！",      # 感叹号
    "？",      # 问号
    "；",      # 分号
    "：",      # 冒号
    "，",      # 逗号
    "、",      # 顿号
    " ",       # 空格
    ""         # 字符（兜底）
]


def get_sample_documents() -> str:
    """获取测试用的中文文档"""
    return """RAG（检索增强生成）是一种结合信息检索和文本生成的人工智能技术。它通过检索外部知识库中的相关信息，为大语言模型提供更准确的上下文，从而生成更可靠的回答。RAG技术有效减少了大模型的幻觉问题。

RAG系统的核心组件包括：文档处理模块、向量化模块、向量数据库和生成模块。文档处理负责将原始文档进行清洗和分块；向量化将文本转换为高维向量；向量数据库存储和检索向量；生成模块利用检索到的上下文生成答案。

向量数据库是RAG系统中存储和检索文本向量的核心组件。常见的向量数据库包括Chroma、FAISS、Milvus、Pinecone、Weaviate等。Chroma是一个轻量级的开源向量数据库，适合快速原型开发。FAISS是Facebook开发的高效向量检索库，支持海量数据。

Embedding模型负责将文本转换为向量表示。中文场景常用的Embedding模型包括BGE系列、M3E系列、text2vec-chinese等。BGE-large-zh是北京智源研究院开发的中文Embedding模型，在多个中文语义理解任务上表现优异，最大支持512个token输入。

文档分块是RAG系统中非常关键的环节。分块策略直接影响检索的准确性和生成的质量。常见的分块策略包括：固定大小分块、基于句子分块、基于段落分块、语义分块等。分块大小需要根据Embedding模型的token限制来设置。

LangChain是一个流行的大模型应用开发框架，提供了丰富的工具来构建RAG系统。它支持多种文档加载器、文本分割器、向量数据库和LLM集成。使用LangChain可以快速搭建RAG应用原型。

检索策略对RAG效果有重要影响。常见的检索策略包括：相似度检索、混合检索（结合关键词和语义）、重排序、多路召回等。选择合适的检索策略可以显著提升回答质量。

RAG的评估指标包括检索准确率、回答相关性、回答准确性、响应延迟等。可以使用RAGAS等框架进行自动化评估。评估结果有助于优化RAG系统的各个组件。

大模型幻觉是指模型生成看似合理但实际错误的内容。RAG通过引入外部知识库，让模型的回答有据可依，有效减少了幻觉问题。这是RAG技术的核心价值之一。

知识库的质量直接影响RAG系统的效果。高质量的知识库应该具备：内容准确、更新及时、覆盖全面、结构清晰等特点。定期维护和更新知识库是保持RAG系统效果的关键。"""


def get_test_queries() -> List[Dict[str, str]]:
    """获取测试查询和期望答案"""
    return [
        {
            "query": "什么是RAG技术？",
            "expected_keywords": ["检索增强生成", "信息检索", "文本生成", "知识库"],
            "category": "基础概念"
        },
        {
            "query": "常见的向量数据库有哪些？",
            "expected_keywords": ["Chroma", "FAISS", "Milvus", "Pinecone"],
            "category": "组件介绍"
        },
        {
            "query": "BGE模型的特点是什么？",
            "expected_keywords": ["BGE", "中文", "Embedding", "512", "token"],
            "category": "模型相关"
        },
        {
            "query": "如何评估RAG系统的效果？",
            "expected_keywords": ["评估", "准确率", "RAGAS", "相关性"],
            "category": "评估方法"
        },
        {
            "query": "分块策略有哪些？",
            "expected_keywords": ["固定大小", "句子", "段落", "语义"],
            "category": "技术细节"
        },
        {
            "query": "RAG如何解决大模型幻觉问题？",
            "expected_keywords": ["幻觉", "知识库", "有据可依", "减少"],
            "category": "核心价值"
        },
    ]


def split_text_with_config(text: str, config: ChunkConfig) -> List[str]:
    """使用指定配置分割文本"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=CHINESE_SEPARATORS,
        length_function=len,
        keep_separator=True
    )

    return splitter.split_text(text)


class ChunkingComparison:
    """分块策略对比测试类"""

    def __init__(self, persist_directory: str = "./chroma_comparison_db"):
        """
        初始化测试类

        Args:
            persist_directory: ChromaDB 持久化目录
        """
        self.persist_directory = persist_directory
        self.embedding_model = None
        self.chroma_client = None
        self.collections: Dict[str, Any] = {}

    def setup(self):
        """初始化 Embedding 模型和 ChromaDB"""
        print("=" * 70)
        print("🚀 初始化测试环境")
        print("=" * 70)

        # 清理旧数据
        if os.path.exists(self.persist_directory):
            print(f"   清理旧数据目录: {self.persist_directory}")
            shutil.rmtree(self.persist_directory)

        # 初始化 Embedding 模型
        print("   加载 Embedding 模型: BAAI/bge-large-zh-v1.5")
        print("   (从 ModelScope 下载模型，首次加载需要耐心等待...)")

        try:
            # 方式1: 从 ModelScope 下载模型
            try:
                from modelscope import snapshot_download
                print("   正在从 ModelScope 下载模型...")
                
                # 从 ModelScope 下载模型到本地缓存
                model_dir = snapshot_download('BAAI/bge-large-zh-v1.5', local_dir="./models")
                print(f"   ✅ 模型下载完成: {model_dir}")
                
                # 检查模型文件类型
                import os
                model_files = os.listdir(model_dir)
                print(f"   📦 模型文件: {[f for f in model_files if f.endswith(('.bin', '.safetensors'))]}")
                
                # 使用手动加载方式加载 pytorch_model.bin
                if 'pytorch_model.bin' in model_files:
                    print(f"   🔄 检测到 pytorch_model.bin 格式，使用手动加载...")
                    self.embedding_model = self._load_from_pytorch_bin(model_dir)
                    print("   ✅ Embedding 模型加载完成 (from ModelScope - pytorch_model.bin)")
                else:
                    # 直接使用 SentenceTransformer 加载
                    from sentence_transformers import SentenceTransformer
                    self.embedding_model = SentenceTransformer(model_dir)
                    print("   ✅ Embedding 模型加载完成 (from ModelScope)")
                
            except ImportError:
                print("   ⚠️ modelscope 库未安装，尝试使用 Hugging Face...")
                raise Exception("请先安装: pip install modelscope")
            except Exception as e:
                print(f"   ⚠️ ModelScope 加载失败: {e}")
                raise
                
        except Exception as e:
            print(f"   ⚠️ 模型加载失败: {e}")
            print("   尝试使用备选模型: BAAI/bge-base-zh-v1.5")
            
            try:
                from modelscope import snapshot_download
                model_dir = snapshot_download('BAAI/bge-base-zh-v1.5', local_dir="./models")
                
                # 检查是否需要手动加载
                import os
                if 'pytorch_model.bin' in os.listdir(model_dir):
                    self.embedding_model = self._load_from_pytorch_bin(model_dir)
                else:
                    from sentence_transformers import SentenceTransformer
                    self.embedding_model = SentenceTransformer(model_dir)
                    
                print("   ✅ 备选模型加载完成 (from ModelScope)")
            except Exception as e2:
                print(f"   ⚠️ 备选模型也加载失败: {e2}")
                # 最终回退到 Hugging Face
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
                print("   ✅ 使用 Hugging Face 加载备选模型")
    
    def _load_from_pytorch_bin(self, model_dir: str):
        """
        手动加载 pytorch_model.bin 格式的模型
        先转换为 safetensors 格式再加载
        """
        import torch
        import json
        from pathlib import Path
        
        print(f"   🔄 正在转换 pytorch_model.bin 到 safetensors 格式...")
        
        # 检查是否已经有 safetensors 文件
        safetensors_path = Path(model_dir) / "model.safetensors"
        if safetensors_path.exists():
            print(f"   ✅ 发现已转换的 safetensors 文件")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_dir)
            return model
        
        # 转换 pytorch_model.bin 到 safetensors
        bin_path = Path(model_dir) / "pytorch_model.bin"
        
        # 加载配置
        with open(f"{model_dir}/config.json", 'r') as f:
            config = json.load(f)
        
        # 加载权重（使用 weights_only=False 绕过检查）
        print(f"   📥 读取 pytorch_model.bin 权重...")
        state_dict = torch.load(
            str(bin_path),
            map_location='cpu',
            weights_only=False
        )
        
        # 保存为 safetensors 格式
        try:
            from safetensors.torch import save_file
            save_file(state_dict, str(safetensors_path))
            print(f"   ✅ 成功转换为 safetensors 格式: {safetensors_path}")
        except ImportError:
            print(f"   ⚠️ safetensors 库未安装，尝试安装...")
            import subprocess
            subprocess.check_call(["pip", "install", "safetensors"])
            from safetensors.torch import save_file
            save_file(state_dict, str(safetensors_path))
            print(f"   ✅ 成功转换为 safetensors 格式: {safetensors_path}")
        
        # 使用 safetensors 格式加载模型
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_dir)
        print(f"   ✅ 模型加载成功")
        
        return model

        # 初始化 ChromaDB
        print("   初始化 ChromaDB...")
        import chromadb
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        print("   ✅ ChromaDB 初始化完成")
        print()

    def create_collections(self, configs: List[ChunkConfig], text: str):
        """为每个配置创建 Collection 并写入数据"""
        print("=" * 70)
        print("📦 创建 Collections 并写入数据")
        print("=" * 70)

        for config in configs:
            print(f"\n🔹 {config.description}")
            print("-" * 50)

            # 分块
            chunks = split_text_with_config(text, config)
            print(f"   分块数量: {len(chunks)}")
            print(f"   平均长度: {sum(len(c) for c in chunks) / len(chunks):.1f} 字符")

            # 创建 Collection
            collection = self.chroma_client.create_collection(
                name=config.name,
                metadata={"description": config.description}
            )

            # 向量化
            print("   向量化中...")
            start_time = time.time()
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
            embed_time = time.time() - start_time
            print(f"   向量化耗时: {embed_time:.2f}秒")

            # 写入 ChromaDB
            ids = [f"{config.name}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"chunk_index": i, "chunk_size": len(chunk)} for i, chunk in enumerate(chunks)]

            collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=chunks,
                metadatas=metadatas
            )
            print(f"   ✅ 写入 {len(chunks)} 条记录到 Collection: {config.name}")

            self.collections[config.name] = {
                "collection": collection,
                "config": config,
                "chunks": chunks,
                "chunk_count": len(chunks)
            }

        print()

    def search(self, query: str, n_results: int = 3) -> Dict[str, List[Dict]]:
        """在所有 Collection 中搜索"""
        # 查询向量化
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        results = {}
        for name, data in self.collections.items():
            collection = data["collection"]

            search_result = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "distances", "metadatas"]
            )

            results[name] = {
                "documents": search_result["documents"][0],
                "distances": search_result["distances"][0],
                "metadatas": search_result["metadatas"][0],
                "config": data["config"]
            }

        return results

    def evaluate_results(self, results: Dict, expected_keywords: List[str]) -> Dict[str, Dict]:
        """评估搜索结果"""
        evaluation = {}

        for config_name, result in results.items():
            documents = result["documents"]
            distances = result["distances"]

            # 计算关键词命中率
            all_text = " ".join(documents)
            hits = sum(1 for kw in expected_keywords if kw in all_text)
            keyword_coverage = hits / len(expected_keywords) * 100

            # 计算平均相似度（距离越小越相似）
            avg_distance = sum(distances) / len(distances)
            # 转换为相似度分数（0-100）
            avg_similarity = (1 - avg_distance) * 100 if avg_distance < 1 else 0

            # 计算结果多样性（不同chunk的数量）
            unique_chunks = len(set(documents))

            evaluation[config_name] = {
                "keyword_coverage": keyword_coverage,
                "avg_similarity": avg_similarity,
                "avg_distance": avg_distance,
                "unique_chunks": unique_chunks,
                "top_distance": distances[0] if distances else 1.0,
                "description": result["config"].description
            }

        return evaluation

    def run_comparison_test(self, queries: List[Dict]) -> Dict:
        """运行对比测试"""
        print("=" * 70)
        print("🔍 执行相似度搜索对比测试")
        print("=" * 70)

        all_evaluations = {config.name: {
            "total_keyword_coverage": 0,
            "total_similarity": 0,
            "query_count": 0,
            "best_match_count": 0,
            "description": ""
        } for config in CHUNK_CONFIGS}

        for i, query_info in enumerate(queries, 1):
            query = query_info["query"]
            expected_keywords = query_info["expected_keywords"]
            category = query_info["category"]

            print(f"\n📌 测试 {i}/{len(queries)}: {query}")
            print(f"   类别: {category}")
            print(f"   期望关键词: {', '.join(expected_keywords)}")
            print("-" * 50)

            # 搜索
            results = self.search(query, n_results=3)

            # 评估
            evaluation = self.evaluate_results(results, expected_keywords)

            # 找出最佳配置
            best_config = max(evaluation.items(),
                              key=lambda x: (x[1]["keyword_coverage"], x[1]["avg_similarity"]))

            # 打印结果
            print(f"\n{'配置':<25} {'关键词覆盖':<12} {'相似度':<12} {'Top距离':<12}")
            print("-" * 60)

            for config_name, eval_result in evaluation.items():
                is_best = "⭐" if config_name == best_config[0] else "  "
                print(f"{is_best}{eval_result['description'][:22]:<23} "
                      f"{eval_result['keyword_coverage']:.1f}%{'':<7} "
                      f"{eval_result['avg_similarity']:.1f}%{'':<7} "
                      f"{eval_result['top_distance']:.4f}")

                # 累计统计
                all_evaluations[config_name]["total_keyword_coverage"] += eval_result["keyword_coverage"]
                all_evaluations[config_name]["total_similarity"] += eval_result["avg_similarity"]
                all_evaluations[config_name]["query_count"] += 1
                all_evaluations[config_name]["description"] = eval_result["description"]
                if config_name == best_config[0]:
                    all_evaluations[config_name]["best_match_count"] += 1

            # 显示最佳配置的检索结果
            print(f"\n   最佳配置检索结果 ({best_config[0]}):")
            for j, doc in enumerate(results[best_config[0]]["documents"][:2], 1):
                print(f"   [{j}] {doc[:60]}...")

        return all_evaluations

    def print_summary(self, evaluations: Dict):
        """打印测试总结"""
        print()
        print("=" * 70)
        print("📊 测试结果总结")
        print("=" * 70)

        # 计算平均值
        summary = []
        for config_name, eval_data in evaluations.items():
            query_count = eval_data["query_count"]
            if query_count > 0:
                avg_keyword = eval_data["total_keyword_coverage"] / query_count
                avg_similarity = eval_data["total_similarity"] / query_count
                best_count = eval_data["best_match_count"]

                # 综合得分 = 关键词覆盖(40%) + 相似度(40%) + 最佳匹配次数(20%)
                composite_score = (avg_keyword * 0.4 + avg_similarity * 0.4 +
                                   (best_count / query_count * 100) * 0.2)

                summary.append({
                    "config": config_name,
                    "description": eval_data["description"],
                    "avg_keyword": avg_keyword,
                    "avg_similarity": avg_similarity,
                    "best_count": best_count,
                    "composite_score": composite_score
                })

        # 按综合得分排序
        summary.sort(key=lambda x: x["composite_score"], reverse=True)

        print(f"\n{'排名':<4} {'配置':<28} {'关键词覆盖':<12} {'相似度':<12} {'最佳次数':<10} {'综合得分':<10}")
        print("-" * 80)

        for rank, item in enumerate(summary, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"{medal}{rank:<3} {item['description'][:26]:<27} "
                  f"{item['avg_keyword']:.1f}%{'':<7} "
                  f"{item['avg_similarity']:.1f}%{'':<7} "
                  f"{item['best_count']:<10} "
                  f"{item['composite_score']:.1f}")

        # 推荐配置
        best = summary[0]
        print()
        print("=" * 70)
        print("💡 推荐配置分析")
        print("=" * 70)
        print(f"\n🏆 最佳配置: {best['description']}")
        print(f"   综合得分: {best['composite_score']:.1f}")
        print(f"   平均关键词覆盖率: {best['avg_keyword']:.1f}%")
        print(f"   平均相似度: {best['avg_similarity']:.1f}%")
        print()

        # 各场景推荐
        print("📋 各场景推荐：")
        print("-" * 50)

        # 找出关键词覆盖最高的配置
        best_keyword = max(summary, key=lambda x: x["avg_keyword"])
        # 找出相似度最高的配置
        best_similarity = max(summary, key=lambda x: x["avg_similarity"])

        print(f"   • 精准匹配场景: {best_keyword['description']}")
        print(f"     (关键词覆盖率最高: {best_keyword['avg_keyword']:.1f}%)")
        print()
        print(f"   • 语义相似场景: {best_similarity['description']}")
        print(f"     (相似度最高: {best_similarity['avg_similarity']:.1f}%)")
        print()
        print(f"   • 综合推荐: {best['description']}")
        print(f"     (综合得分最高: {best['composite_score']:.1f})")

    def show_collection_stats(self):
        """显示 Collection 统计信息"""
        print()
        print("=" * 70)
        print("📈 Collection 统计")
        print("=" * 70)

        print(f"\n{'配置':<28} {'Chunk数':<10} {'平均长度':<12} {'配置参数':<20}")
        print("-" * 70)

        for name, data in self.collections.items():
            config = data["config"]
            chunks = data["chunks"]
            avg_len = sum(len(c) for c in chunks) / len(chunks)

            print(f"{config.description[:26]:<28} "
                  f"{len(chunks):<10} "
                  f"{avg_len:.1f}字符{'':<5} "
                  f"size={config.chunk_size}, overlap={config.chunk_overlap}")

    def cleanup(self):
        """清理资源"""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print(f"\n🧹 已清理测试数据目录: {self.persist_directory}")


def main():
    """主函数"""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " 分块策略效果对比测试 - BGE-large-zh + ChromaDB ".center(58) + "   ║")
    print("╚" + "═" * 68 + "╝")
    print()

    # 检查依赖
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("\n请安装依赖：")
        print("pip install langchain-text-splitters chromadb sentence-transformers")
        return

    # 创建测试实例
    comparison = ChunkingComparison()

    try:
        # 1. 初始化环境
        comparison.setup()

        # 2. 获取测试数据
        sample_text = get_sample_documents()
        test_queries = get_test_queries()

        print(f"📄 测试文档长度: {len(sample_text)} 字符")
        print(f"❓ 测试查询数量: {len(test_queries)} 个")
        print()

        # 3. 创建 Collections
        comparison.create_collections(CHUNK_CONFIGS, sample_text)

        # 4. 显示统计信息
        comparison.show_collection_stats()

        # 5. 运行对比测试
        evaluations = comparison.run_comparison_test(test_queries)

        # 6. 打印总结
        comparison.print_summary(evaluations)

        print()
        print("=" * 70)
        print("✅ 测试完成！")
        print("=" * 70)
        print()
        print("📝 结论说明：")
        print("   • 关键词覆盖率: 检索结果中包含期望关键词的比例")
        print("   • 相似度: 查询与检索结果的向量相似度")
        print("   • 最佳次数: 在所有测试中获得最佳结果的次数")
        print("   • 综合得分: 关键词覆盖(40%) + 相似度(40%) + 最佳比例(20%)")
        print()
        print("🔧 根据测试结果选择适合您场景的分块配置！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 询问是否清理
        print()
        try:
            cleanup = input("是否清理测试数据？(y/n, 默认y): ").strip().lower()
            if cleanup != 'n':
                comparison.cleanup()
        except EOFError:
            # 非交互模式下自动清理
            comparison.cleanup()


if __name__ == "__main__":
    main()
