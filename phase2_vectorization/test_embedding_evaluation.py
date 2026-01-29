#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding 模型效果评估脚本

本脚本用于评估不同 Embedding 模型在中文 RAG 场景下的效果：
1. 检索质量指标：Hit Rate、MRR、NDCG、Recall@K、Precision@K
2. 语义理解能力：正负样本区分度
3. 推理性能：向量化速度和吞吐量
4. 可视化分析：生成对比图表

依赖安装：
pip install sentence-transformers numpy matplotlib scikit-learn
"""

import os
import sys
import time
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np


# ============================================================
# 配置定义
# ============================================================
@dataclass
class EmbeddingModelConfig:
    """Embedding 模型配置"""
    name: str
    model_id: str
    description: str
    max_seq_length: int


# 待评估的 Embedding 模型列表
EMBEDDING_MODELS = [
    EmbeddingModelConfig(
        name="bge-large-zh",
        model_id="BAAI/bge-large-zh-v1.5",
        description="BGE-large-zh (北京智源, 1024维)",
        max_seq_length=512
    ),
    EmbeddingModelConfig(
        name="bge-base-zh",
        model_id="BAAI/bge-base-zh-v1.5",
        description="BGE-base-zh (北京智源, 768维)",
        max_seq_length=512
    ),
    EmbeddingModelConfig(
        name="text2vec-base",
        model_id="shibing624/text2vec-base-chinese",
        description="text2vec-base-chinese (768维)",
        max_seq_length=128
    ),
    EmbeddingModelConfig(
        name="m3e-base",
        model_id="moka-ai/m3e-base",
        description="M3E-base (Moka AI, 768维)",
        max_seq_length=512
    ),
]


# ============================================================
# 测试数据集
# ============================================================
def get_evaluation_dataset() -> Dict[str, Any]:
    """
    获取评估数据集
    
    数据集包含：
    1. 文档库：用于构建检索库
    2. 查询集：包含查询和对应的相关文档ID
    3. 正负样本对：用于测试语义区分能力
    """
    
    # 文档库（模拟知识库中的文档片段）
    documents = {
        "doc_1": "RAG是检索增强生成技术，通过检索外部知识库为大语言模型提供上下文，有效减少幻觉问题。",
        "doc_2": "向量数据库是RAG系统的核心组件，常见的有Chroma、FAISS、Milvus、Pinecone等。",
        "doc_3": "BGE是北京智源研究院开发的中文Embedding模型，在语义理解任务上表现优异。",
        "doc_4": "文档分块策略直接影响RAG检索效果，常见策略包括固定大小分块和语义分块。",
        "doc_5": "LangChain是构建RAG应用的流行框架，提供文档加载、分割、向量化等工具。",
        "doc_6": "大模型幻觉是指模型生成看似合理但实际错误的内容，RAG可以有效缓解这个问题。",
        "doc_7": "Embedding模型将文本转换为高维向量，使得语义相似的文本在向量空间中距离更近。",
        "doc_8": "检索策略包括相似度检索、混合检索、重排序等，选择合适的策略可提升效果。",
        "doc_9": "RAGAS是一个RAG评估框架，可以自动评估检索准确率和回答质量。",
        "doc_10": "知识库质量直接影响RAG效果，需要保证内容准确、更新及时、覆盖全面。",
        "doc_11": "Python是人工智能领域最流行的编程语言，拥有丰富的机器学习库。",
        "doc_12": "深度学习是机器学习的一个分支，使用神经网络进行特征学习和模式识别。",
        "doc_13": "自然语言处理（NLP）是让计算机理解和生成人类语言的技术领域。",
        "doc_14": "Transformer架构是现代大语言模型的基础，引入了自注意力机制。",
        "doc_15": "GPT和BERT是两种重要的预训练语言模型架构，分别采用自回归和双向编码。",
    }
    
    # 查询集（每个查询对应的相关文档ID列表）
    queries = [
        {
            "query": "什么是RAG技术？它有什么作用？",
            "relevant_docs": ["doc_1", "doc_6"],
            "category": "基础概念"
        },
        {
            "query": "有哪些常用的向量数据库？",
            "relevant_docs": ["doc_2"],
            "category": "技术组件"
        },
        {
            "query": "BGE模型是什么？",
            "relevant_docs": ["doc_3", "doc_7"],
            "category": "模型相关"
        },
        {
            "query": "如何对文档进行分块？",
            "relevant_docs": ["doc_4"],
            "category": "技术细节"
        },
        {
            "query": "LangChain框架的功能是什么？",
            "relevant_docs": ["doc_5"],
            "category": "工具框架"
        },
        {
            "query": "如何评估RAG系统效果？",
            "relevant_docs": ["doc_9", "doc_10"],
            "category": "评估方法"
        },
        {
            "query": "什么是大模型幻觉？如何解决？",
            "relevant_docs": ["doc_6", "doc_1"],
            "category": "问题解决"
        },
        {
            "query": "检索策略有哪些类型？",
            "relevant_docs": ["doc_8"],
            "category": "技术细节"
        },
        {
            "query": "Embedding模型的作用是什么？",
            "relevant_docs": ["doc_7", "doc_3"],
            "category": "模型相关"
        },
        {
            "query": "Transformer架构的特点是什么？",
            "relevant_docs": ["doc_14", "doc_15"],
            "category": "深度学习"
        },
    ]
    
    # 正负样本对（用于测试语义区分能力）
    # 格式：(文本1, 文本2, 标签) 标签1=相似，0=不相似
    semantic_pairs = [
        # 正样本对（语义相似）
        ("RAG技术可以减少大模型幻觉", "检索增强生成有助于提升回答准确性", 1),
        ("向量数据库存储文本向量", "Embedding向量被保存在向量数据库中", 1),
        ("BGE是中文Embedding模型", "北京智源的BGE模型用于文本向量化", 1),
        ("文档分块影响检索效果", "合理的分块策略可以提升RAG准确率", 1),
        ("LangChain用于构建RAG应用", "使用LangChain框架开发检索增强系统", 1),
        ("自然语言处理是AI领域", "NLP让机器理解人类语言", 1),
        ("深度学习使用神经网络", "神经网络是深度学习的核心", 1),
        ("Transformer引入注意力机制", "自注意力是Transformer的关键创新", 1),
        
        # 负样本对（语义不相似）
        ("RAG技术用于增强检索", "今天天气很好适合出门", 0),
        ("向量数据库存储向量", "我喜欢吃苹果和香蕉", 0),
        ("BGE是Embedding模型", "北京是中国的首都城市", 0),
        ("文档分块策略很重要", "篮球是一项团队运动", 0),
        ("LangChain是开发框架", "长江是中国最长的河流", 0),
        ("机器学习是AI分支", "音乐可以陶冶情操", 0),
        ("Python是编程语言", "熊猫是中国国宝动物", 0),
        ("Transformer架构很重要", "咖啡有提神的作用", 0),
    ]
    
    return {
        "documents": documents,
        "queries": queries,
        "semantic_pairs": semantic_pairs
    }


# ============================================================
# 评估指标计算
# ============================================================
class EvaluationMetrics:
    """评估指标计算类"""
    
    @staticmethod
    def hit_rate(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        计算 Hit Rate @K
        如果Top-K结果中包含任一相关文档，则命中
        
        Args:
            retrieved_ids: 检索返回的文档ID列表
            relevant_ids: 相关文档ID列表
            k: Top-K
        
        Returns:
            1.0 如果命中，否则 0.0
        """
        top_k = retrieved_ids[:k]
        return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0
    
    @staticmethod
    def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        计算 MRR (Mean Reciprocal Rank)
        第一个相关文档排名的倒数
        
        Args:
            retrieved_ids: 检索返回的文档ID列表
            relevant_ids: 相关文档ID列表
        
        Returns:
            1/rank 或 0.0（如果没有命中）
        """
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0
    
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        计算 Precision @K
        Top-K结果中相关文档的比例
        
        Args:
            retrieved_ids: 检索返回的文档ID列表
            relevant_ids: 相关文档ID列表
            k: Top-K
        
        Returns:
            相关文档数 / K
        """
        top_k = retrieved_ids[:k]
        relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return relevant_count / k
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        计算 Recall @K
        Top-K结果中召回的相关文档比例
        
        Args:
            retrieved_ids: 检索返回的文档ID列表
            relevant_ids: 相关文档ID列表
            k: Top-K
        
        Returns:
            召回的相关文档数 / 总相关文档数
        """
        top_k = retrieved_ids[:k]
        recalled = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return recalled / len(relevant_ids) if relevant_ids else 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        计算 NDCG @K (Normalized Discounted Cumulative Gain)
        考虑排序位置的相关性得分
        
        Args:
            retrieved_ids: 检索返回的文档ID列表
            relevant_ids: 相关文档ID列表
            k: Top-K
        
        Returns:
            NDCG 分数
        """
        def dcg(relevances: List[int]) -> float:
            """计算 DCG"""
            return sum(
                rel / np.log2(idx + 2)  # idx+2 因为 log2(1) = 0
                for idx, rel in enumerate(relevances)
            )
        
        # 实际相关性列表
        actual_relevances = [
            1 if doc_id in relevant_ids else 0
            for doc_id in retrieved_ids[:k]
        ]
        
        # 理想相关性列表（所有相关文档排在前面）
        ideal_relevances = sorted(actual_relevances, reverse=True)
        
        actual_dcg = dcg(actual_relevances)
        ideal_dcg = dcg(ideal_relevances)
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0


# ============================================================
# Embedding 模型评估器
# ============================================================
class EmbeddingEvaluator:
    """Embedding 模型评估器"""
    
    def __init__(self, persist_directory: str = "./chroma_eval_db"):
        """
        初始化评估器
        
        Args:
            persist_directory: ChromaDB 持久化目录
        """
        self.persist_directory = persist_directory
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        self.metrics = EvaluationMetrics()
        
    def load_model(self, config: EmbeddingModelConfig) -> Optional[Any]:
        """
        加载 Embedding 模型
        
        Args:
            config: 模型配置
            
        Returns:
            加载的模型或 None
        """
        try:
            print(f"   加载模型: {config.model_id}")
            print(f"   (从 ModelScope 下载模型，首次加载需要耐心等待...)")
            
            # 优先从 ModelScope 下载模型
            try:
                from modelscope import snapshot_download
                print(f"   正在从 ModelScope 下载模型: {config.model_id}...")
                
                # 从 ModelScope 下载模型到本地缓存
                model_dir = snapshot_download(config.model_id, cache_dir="./model_cache")
                print(f"   ✅ 模型下载完成: {model_dir}")
                
                # 检查模型文件类型
                import os
                model_files = os.listdir(model_dir)
                print(f"   📦 模型文件: {[f for f in model_files if f.endswith(('.bin', '.safetensors'))]}")
                
                # 使用手动加载方式加载 pytorch_model.bin
                if 'pytorch_model.bin' in model_files:
                    print(f"   🔄 检测到 pytorch_model.bin 格式，使用手动加载...")
                    model = self._load_from_pytorch_bin(model_dir)
                    if model is not None:
                        print(f"   ✅ 模型加载成功 (from ModelScope - pytorch_model.bin): {config.name}")
                        return model
                    print(f"   ⚠️ 手动加载失败，尝试 SentenceTransformer 直接加载...")
                
                # 如果有 safetensors，直接加载
                elif any(f.endswith('.safetensors') for f in model_files):
                    print(f"   🔄 检测到 safetensors 格式，使用 SentenceTransformer 加载...")
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(model_dir)
                    print(f"   ✅ 模型加载成功 (from ModelScope - safetensors): {config.name}")
                    return model
                
                # 尝试直接使用 SentenceTransformer 加载
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_dir)
                print(f"   ✅ 模型加载成功 (from ModelScope): {config.name}")
                return model
                
            except ImportError:
                print("   ⚠️ modelscope 库未安装，尝试使用 Hugging Face...")
                # 继续尝试下面的 Hugging Face 方式
            except Exception as e:
                print(f"   ⚠️ ModelScope 加载失败: {str(e)[:100]}")
                print("   尝试使用 Hugging Face...")
            
            # 如果 ModelScope 失败，尝试使用 Hugging Face
            from sentence_transformers import SentenceTransformer
            
            # 方式1: 直接加载（trust_remote_code解决某些模型兼容性问题）
            try:
                model = SentenceTransformer(
                    config.model_id,
                    trust_remote_code=True
                )
                print(f"   ✅ 模型加载成功 (from Hugging Face): {config.name}")
                return model
            except Exception as e1:
                print(f"   ⚠️ 直接加载失败: {str(e1)[:100]}")
            
            # 方式2: 禁用 safetensors
            try:
                model = SentenceTransformer(
                    config.model_id,
                    trust_remote_code=True,
                    model_kwargs={"use_safetensors": False}
                )
                print(f"   ✅ 模型加载成功 (禁用safetensors): {config.name}")
                return model
            except Exception as e2:
                print(f"   ⚠️ 禁用safetensors加载失败: {str(e2)[:100]}")
            
            return None
            
        except Exception as e:
            print(f"   ❌ 模型加载失败: {config.name} - {e}")
            return None
    
    def _load_from_pytorch_bin(self, model_dir: str) -> Optional[Any]:
        """
        手动加载 pytorch_model.bin 格式的模型
        先转换为 safetensors 格式再加载
        
        Args:
            model_dir: 模型目录路径
            
        Returns:
            加载的模型或 None
        """
        try:
            import torch
            import json
            from pathlib import Path
            
            # 检查是否已经有 safetensors 文件
            safetensors_path = Path(model_dir) / "model.safetensors"
            if safetensors_path.exists():
                print(f"   ✅ 发现已转换的 safetensors 文件")
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_dir)
                return model
            
            # 转换 pytorch_model.bin 到 safetensors
            print(f"   🔄 正在转换 pytorch_model.bin 到 safetensors 格式...")
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
            
            # 保存为 safetensors 格式 将模型model.bin文件转为safetensors格式
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
            
        except Exception as e:
            print(f"   ❌ 手动加载失败: {str(e)[:150]}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_retrieval(
        self,
        model: Any,
        model_name: str,
        documents: Dict[str, str],
        queries: List[Dict],
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, Any]:
        """
        评估检索质量
        
        Args:
            model: Embedding 模型
            model_name: 模型名称
            documents: 文档库
            queries: 查询集
            k_values: K值列表
            
        Returns:
            评估结果
        """
        print(f"\n   📊 评估检索质量...")
        
        # 构建文档向量库
        doc_ids = list(documents.keys())
        doc_texts = list(documents.values())
        
        # 向量化文档
        start_time = time.time()
        doc_embeddings = model.encode(doc_texts, show_progress_bar=False)
        doc_encode_time = time.time() - start_time
        
        # 初始化指标累计
        metrics_sum = {
            f"hit_rate@{k}": 0.0 for k in k_values
        }
        metrics_sum.update({
            f"precision@{k}": 0.0 for k in k_values
        })
        metrics_sum.update({
            f"recall@{k}": 0.0 for k in k_values
        })
        metrics_sum.update({
            f"ndcg@{k}": 0.0 for k in k_values
        })
        metrics_sum["mrr"] = 0.0
        
        query_results = []
        
        # 对每个查询进行评估
        for query_info in queries:
            query = query_info["query"]
            relevant_docs = query_info["relevant_docs"]
            
            # 查询向量化
            query_embedding = model.encode([query], show_progress_bar=False)[0]
            
            # 计算相似度并排序
            similarities = []
            for idx, doc_emb in enumerate(doc_embeddings):
                sim = self.metrics.cosine_similarity(query_embedding, doc_emb)
                similarities.append((doc_ids[idx], sim))
            
            # 按相似度降序排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            retrieved_ids = [doc_id for doc_id, _ in similarities]
            
            # 计算各项指标
            query_metrics = {}
            for k in k_values:
                query_metrics[f"hit_rate@{k}"] = self.metrics.hit_rate(retrieved_ids, relevant_docs, k)
                query_metrics[f"precision@{k}"] = self.metrics.precision_at_k(retrieved_ids, relevant_docs, k)
                query_metrics[f"recall@{k}"] = self.metrics.recall_at_k(retrieved_ids, relevant_docs, k)
                query_metrics[f"ndcg@{k}"] = self.metrics.ndcg_at_k(retrieved_ids, relevant_docs, k)
                
                metrics_sum[f"hit_rate@{k}"] += query_metrics[f"hit_rate@{k}"]
                metrics_sum[f"precision@{k}"] += query_metrics[f"precision@{k}"]
                metrics_sum[f"recall@{k}"] += query_metrics[f"recall@{k}"]
                metrics_sum[f"ndcg@{k}"] += query_metrics[f"ndcg@{k}"]
            
            mrr = self.metrics.mrr(retrieved_ids, relevant_docs)
            query_metrics["mrr"] = mrr
            metrics_sum["mrr"] += mrr
            
            query_results.append({
                "query": query,
                "relevant_docs": relevant_docs,
                "top_3_retrieved": retrieved_ids[:3],
                "top_3_scores": [s for _, s in similarities[:3]],
                "metrics": query_metrics
            })
        
        # 计算平均值
        num_queries = len(queries)
        avg_metrics = {k: v / num_queries for k, v in metrics_sum.items()}
        
        return {
            "avg_metrics": avg_metrics,
            "query_results": query_results,
            "doc_encode_time": doc_encode_time,
            "num_documents": len(documents),
            "num_queries": num_queries
        }
    
    def evaluate_semantic_discrimination(
        self,
        model: Any,
        model_name: str,
        semantic_pairs: List[Tuple[str, str, int]]
    ) -> Dict[str, Any]:
        """
        评估语义区分能力
        
        Args:
            model: Embedding 模型
            model_name: 模型名称
            semantic_pairs: 正负样本对列表
            
        Returns:
            评估结果
        """
        print(f"   📊 评估语义区分能力...")
        
        positive_similarities = []
        negative_similarities = []
        predictions = []
        labels = []
        
        for text1, text2, label in semantic_pairs:
            # 计算相似度
            emb1 = model.encode([text1], show_progress_bar=False)[0]
            emb2 = model.encode([text2], show_progress_bar=False)[0]
            similarity = self.metrics.cosine_similarity(emb1, emb2)
            
            if label == 1:
                positive_similarities.append(similarity)
            else:
                negative_similarities.append(similarity)
            
            labels.append(label)
            predictions.append(similarity)
        
        # 计算统计指标
        avg_positive_sim = np.mean(positive_similarities) if positive_similarities else 0
        avg_negative_sim = np.mean(negative_similarities) if negative_similarities else 0
        discrimination_gap = avg_positive_sim - avg_negative_sim
        
        # 计算 AUC（使用相似度作为预测分数）
        try:
            from sklearn.metrics import roc_auc_score
            auc_score = roc_auc_score(labels, predictions)
        except Exception:
            auc_score = 0.0
        
        # 使用阈值计算准确率（阈值 = 正负样本平均相似度的中点）
        threshold = (avg_positive_sim + avg_negative_sim) / 2
        correct = sum(
            1 for pred, label in zip(predictions, labels)
            if (pred >= threshold and label == 1) or (pred < threshold and label == 0)
        )
        accuracy = correct / len(labels) if labels else 0
        
        return {
            "avg_positive_similarity": avg_positive_sim,
            "avg_negative_similarity": avg_negative_sim,
            "discrimination_gap": discrimination_gap,
            "auc_score": auc_score,
            "accuracy": accuracy,
            "threshold": threshold,
            "positive_similarities": positive_similarities,
            "negative_similarities": negative_similarities
        }
    
    def evaluate_performance(
        self,
        model: Any,
        model_name: str,
        test_texts: List[str],
        batch_sizes: List[int] = [1, 8, 32]
    ) -> Dict[str, Any]:
        """
        评估推理性能
        
        Args:
            model: Embedding 模型
            model_name: 模型名称
            test_texts: 测试文本列表
            batch_sizes: 批次大小列表
            
        Returns:
            评估结果
        """
        print(f"   📊 评估推理性能...")
        
        performance_results = {}
        
        for batch_size in batch_sizes:
            # 准备测试数据
            num_batches = len(test_texts) // batch_size
            if num_batches == 0:
                continue
            
            total_time = 0
            total_texts = 0
            
            for i in range(num_batches):
                batch = test_texts[i * batch_size: (i + 1) * batch_size]
                
                start_time = time.time()
                _ = model.encode(batch, show_progress_bar=False)
                elapsed = time.time() - start_time
                
                total_time += elapsed
                total_texts += len(batch)
            
            avg_latency = total_time / num_batches * 1000  # 毫秒
            throughput = total_texts / total_time if total_time > 0 else 0  # texts/sec
            
            performance_results[f"batch_{batch_size}"] = {
                "avg_latency_ms": avg_latency,
                "throughput": throughput,
                "total_texts": total_texts,
                "total_time": total_time
            }
        
        return performance_results
    
    def run_full_evaluation(self, model_configs: List[EmbeddingModelConfig] = None):
        """
        运行完整评估
        
        Args:
            model_configs: 模型配置列表（默认使用 EMBEDDING_MODELS）
        """
        if model_configs is None:
            model_configs = EMBEDDING_MODELS
        
        # 获取评估数据集
        dataset = get_evaluation_dataset()
        documents = dataset["documents"]
        queries = dataset["queries"]
        semantic_pairs = dataset["semantic_pairs"]
        
        print("=" * 70)
        print("🚀 Embedding 模型效果评估")
        print("=" * 70)
        print(f"\n📄 数据集统计:")
        print(f"   文档数量: {len(documents)}")
        print(f"   查询数量: {len(queries)}")
        print(f"   语义对数量: {len(semantic_pairs)}")
        print(f"   待评估模型: {len(model_configs)} 个")
        
        # 性能测试用的文本
        perf_test_texts = list(documents.values()) * 3  # 45个文本
        
        # 逐个评估模型
        for config in model_configs:
            print()
            print("=" * 70)
            print(f"🔹 评估模型: {config.description}")
            print("=" * 70)
            
            # 加载模型
            model = self.load_model(config)
            if model is None:
                print(f"   ⚠️ 跳过模型: {config.name}")
                self.results[config.name] = {"status": "failed", "error": "模型加载失败"}
                continue
            
            self.models[config.name] = model
            
            # 1. 检索质量评估
            retrieval_results = self.evaluate_retrieval(
                model, config.name, documents, queries
            )
            
            # 2. 语义区分能力评估
            semantic_results = self.evaluate_semantic_discrimination(
                model, config.name, semantic_pairs
            )
            
            # 3. 推理性能评估
            performance_results = self.evaluate_performance(
                model, config.name, perf_test_texts
            )
            
            # 保存结果
            self.results[config.name] = {
                "status": "success",
                "config": config,
                "retrieval": retrieval_results,
                "semantic": semantic_results,
                "performance": performance_results
            }
            
            # 打印简要结果
            self._print_model_summary(config.name)
        
        # 打印总结对比
        self._print_comparison_summary()
    
    def _print_model_summary(self, model_name: str):
        """打印单个模型的评估摘要"""
        result = self.results.get(model_name)
        if not result or result.get("status") != "success":
            return
        
        retrieval = result["retrieval"]["avg_metrics"]
        semantic = result["semantic"]
        
        print(f"\n   📈 评估结果摘要:")
        print(f"   {'─' * 40}")
        print(f"   检索指标:")
        print(f"      Hit Rate@1: {retrieval.get('hit_rate@1', 0):.1%}")
        print(f"      Hit Rate@3: {retrieval.get('hit_rate@3', 0):.1%}")
        print(f"      MRR: {retrieval.get('mrr', 0):.3f}")
        print(f"      NDCG@3: {retrieval.get('ndcg@3', 0):.3f}")
        print(f"   语义区分:")
        print(f"      正样本平均相似度: {semantic['avg_positive_similarity']:.3f}")
        print(f"      负样本平均相似度: {semantic['avg_negative_similarity']:.3f}")
        print(f"      区分度: {semantic['discrimination_gap']:.3f}")
        print(f"      AUC: {semantic['auc_score']:.3f}")
    
    def _print_comparison_summary(self):
        """打印模型对比总结"""
        print()
        print("=" * 70)
        print("📊 模型对比总结")
        print("=" * 70)
        
        # 收集成功的模型结果
        valid_results = [
            (name, result) for name, result in self.results.items()
            if result.get("status") == "success"
        ]
        
        if not valid_results:
            print("\n⚠️ 没有成功评估的模型")
            return
        
        # 1. 检索质量对比
        print("\n1️⃣ 检索质量对比")
        print("-" * 70)
        print(f"{'模型':<25} {'Hit@1':<10} {'Hit@3':<10} {'MRR':<10} {'NDCG@3':<10}")
        print("-" * 70)
        
        retrieval_scores = []
        for name, result in valid_results:
            metrics = result["retrieval"]["avg_metrics"]
            hit1 = metrics.get("hit_rate@1", 0)
            hit3 = metrics.get("hit_rate@3", 0)
            mrr = metrics.get("mrr", 0)
            ndcg3 = metrics.get("ndcg@3", 0)
            
            # 综合得分
            composite = hit1 * 0.3 + hit3 * 0.2 + mrr * 0.3 + ndcg3 * 0.2
            retrieval_scores.append((name, composite))
            
            print(f"{name:<25} {hit1:<10.1%} {hit3:<10.1%} {mrr:<10.3f} {ndcg3:<10.3f}")
        
        # 2. 语义区分能力对比
        print("\n2️⃣ 语义区分能力对比")
        print("-" * 70)
        print(f"{'模型':<25} {'正样本相似度':<15} {'负样本相似度':<15} {'区分度':<10} {'AUC':<10}")
        print("-" * 70)
        
        semantic_scores = []
        for name, result in valid_results:
            sem = result["semantic"]
            pos = sem["avg_positive_similarity"]
            neg = sem["avg_negative_similarity"]
            gap = sem["discrimination_gap"]
            auc = sem["auc_score"]
            
            semantic_scores.append((name, auc))
            print(f"{name:<25} {pos:<15.3f} {neg:<15.3f} {gap:<10.3f} {auc:<10.3f}")
        
        # 3. 推理性能对比
        print("\n3️⃣ 推理性能对比 (batch_size=8)")
        print("-" * 70)
        print(f"{'模型':<25} {'延迟(ms)':<15} {'吞吐量(texts/s)':<20}")
        print("-" * 70)
        
        performance_scores = []
        for name, result in valid_results:
            perf = result["performance"]
            if "batch_8" in perf:
                latency = perf["batch_8"]["avg_latency_ms"]
                throughput = perf["batch_8"]["throughput"]
                performance_scores.append((name, throughput))
                print(f"{name:<25} {latency:<15.1f} {throughput:<20.1f}")
            else:
                print(f"{name:<25} {'N/A':<15} {'N/A':<20}")
        
        # 4. 综合评分
        print("\n4️⃣ 综合评分排名")
        print("-" * 70)
        
        # 计算综合得分（检索50% + 语义30% + 性能20%）
        composite_scores = []
        for name, result in valid_results:
            retrieval = result["retrieval"]["avg_metrics"]
            semantic = result["semantic"]
            
            # 检索得分 (归一化到0-1)
            ret_score = (
                retrieval.get("hit_rate@1", 0) * 0.3 +
                retrieval.get("hit_rate@3", 0) * 0.2 +
                retrieval.get("mrr", 0) * 0.3 +
                retrieval.get("ndcg@3", 0) * 0.2
            )
            
            # 语义得分
            sem_score = semantic["auc_score"]
            
            # 性能得分（相对分数）
            perf = result["performance"]
            perf_score = 0.5  # 默认中等分数
            if "batch_8" in perf and performance_scores:
                max_throughput = max(s[1] for s in performance_scores if s[1] > 0)
                if max_throughput > 0:
                    perf_score = perf["batch_8"]["throughput"] / max_throughput
            
            # 综合得分
            composite = ret_score * 0.5 + sem_score * 0.3 + perf_score * 0.2
            composite_scores.append((name, composite, ret_score, sem_score, perf_score))
        
        # 按综合得分排序
        composite_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"{'排名':<6} {'模型':<25} {'综合得分':<12} {'检索':<10} {'语义':<10} {'性能':<10}")
        print("-" * 70)
        
        for rank, (name, composite, ret, sem, perf) in enumerate(composite_scores, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"{medal}{rank:<5} {name:<25} {composite:<12.3f} {ret:<10.3f} {sem:<10.3f} {perf:<10.3f}")
        
        # 推荐
        if composite_scores:
            best = composite_scores[0]
            print()
            print("=" * 70)
            print("💡 评估结论")
            print("=" * 70)
            print(f"\n🏆 综合最佳模型: {best[0]}")
            print(f"   综合得分: {best[1]:.3f}")
            print()
            
            # 各维度最佳
            best_retrieval = max(valid_results, key=lambda x: x[1]["retrieval"]["avg_metrics"].get("mrr", 0))
            best_semantic = max(valid_results, key=lambda x: x[1]["semantic"]["auc_score"])
            
            print("📋 各维度最佳:")
            print(f"   • 检索质量最佳: {best_retrieval[0]}")
            print(f"   • 语义区分最佳: {best_semantic[0]}")
            if performance_scores:
                best_perf = max(performance_scores, key=lambda x: x[1])
                print(f"   • 推理速度最快: {best_perf[0]}")

    def generate_visualization(self, output_dir: str = "./eval_results"):
        """
        生成可视化图表
        
        Args:
            output_dir: 输出目录
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
        except ImportError:
            print("\n⚠️ 未安装 matplotlib，跳过可视化生成")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        valid_results = [
            (name, result) for name, result in self.results.items()
            if result.get("status") == "success"
        ]
        
        if not valid_results:
            return
        
        print()
        print("=" * 70)
        print("📈 生成可视化图表")
        print("=" * 70)
        
        # 1. 检索指标对比柱状图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        model_names = [name for name, _ in valid_results]
        
        # 检索指标
        hit1_scores = [r["retrieval"]["avg_metrics"].get("hit_rate@1", 0) for _, r in valid_results]
        hit3_scores = [r["retrieval"]["avg_metrics"].get("hit_rate@3", 0) for _, r in valid_results]
        mrr_scores = [r["retrieval"]["avg_metrics"].get("mrr", 0) for _, r in valid_results]
        
        x = np.arange(len(model_names))
        width = 0.25
        
        axes[0].bar(x - width, hit1_scores, width, label='Hit@1', color='#2ecc71')
        axes[0].bar(x, hit3_scores, width, label='Hit@3', color='#3498db')
        axes[0].bar(x + width, mrr_scores, width, label='MRR', color='#e74c3c')
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Retrieval Quality Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0].legend()
        axes[0].set_ylim(0, 1.1)
        
        # 语义区分能力
        pos_sims = [r["semantic"]["avg_positive_similarity"] for _, r in valid_results]
        neg_sims = [r["semantic"]["avg_negative_similarity"] for _, r in valid_results]
        
        axes[1].bar(x - width/2, pos_sims, width, label='Positive Similarity', color='#27ae60')
        axes[1].bar(x + width/2, neg_sims, width, label='Negative Similarity', color='#c0392b')
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Similarity')
        axes[1].set_title('Semantic Discrimination Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1].legend()
        
        plt.tight_layout()
        chart_path = os.path.join(output_dir, "embedding_comparison.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 保存图表: {chart_path}")
        
        # 2. AUC 指标对比图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        auc_scores = [r["semantic"]["auc_score"] for _, r in valid_results]
        discrimination_gaps = [r["semantic"]["discrimination_gap"] for _, r in valid_results]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, auc_scores, width, label='AUC Score', color='#9b59b6', alpha=0.8)
        bars2 = ax.bar(x + width/2, discrimination_gaps, width, label='Discrimination Gap', color='#f39c12', alpha=0.8)
        
        # 在柱状图上显示数值
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('AUC Score and Discrimination Gap Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Random Guess (0.5)')
        ax.legend()
        
        plt.tight_layout()
        auc_chart_path = os.path.join(output_dir, "auc_comparison.png")
        plt.savefig(auc_chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 保存图表: {auc_chart_path}")
        
        # 3. 相似度分布箱线图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        all_data = []
        all_labels = []
        for name, result in valid_results:
            pos = result["semantic"]["positive_similarities"]
            neg = result["semantic"]["negative_similarities"]
            all_data.extend([pos, neg])
            all_labels.extend([f"{name}\n(Positive)", f"{name}\n(Negative)"])
        
        bp = ax.boxplot(all_data, labels=all_labels, patch_artist=True)
        
        colors = ['#2ecc71', '#e74c3c'] * len(valid_results)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Similarity Distribution by Model')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        boxplot_path = os.path.join(output_dir, "similarity_distribution.png")
        plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 保存图表: {boxplot_path}")
        
        # 4. 综合指标雷达图
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # 准备雷达图数据
        categories = ['Hit@1', 'Hit@3', 'MRR', 'AUC', 'Discrimination Gap']
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合图形
        
        # 为每个模型绘制雷达图
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        for idx, (name, result) in enumerate(valid_results):
            values = [
                result["retrieval"]["avg_metrics"].get("hit_rate@1", 0),
                result["retrieval"]["avg_metrics"].get("hit_rate@3", 0),
                result["retrieval"]["avg_metrics"].get("mrr", 0),
                result["semantic"]["auc_score"],
                result["semantic"]["discrimination_gap"]
            ]
            values += values[:1]  # 闭合图形
            
            color = colors[idx % len(colors)]
            ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1.1)
        ax.set_title('Comprehensive Performance Radar Chart', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        radar_path = os.path.join(output_dir, "radar_comparison.png")
        plt.savefig(radar_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 保存图表: {radar_path}")
        
        print(f"\n📁 图表保存目录: {output_dir}")


def main():
    """主函数"""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " Embedding 模型效果评估工具 ".center(58) + "        ║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # 检查依赖
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("\n请安装依赖：")
        print("pip install sentence-transformers numpy matplotlib scikit-learn")
        sys.exit(1)
    
    # 创建评估器
    evaluator = EmbeddingEvaluator()
    
    try:
        # 运行评估
        evaluator.run_full_evaluation()
        
        # 生成可视化
        evaluator.generate_visualization()
        
        print()
        print("=" * 70)
        print("✅ 评估完成！")
        print("=" * 70)
        print()
        print("📝 指标说明：")
        print("   • Hit Rate@K: Top-K结果中命中相关文档的比例")
        print("   • MRR: 第一个相关文档排名的倒数均值")
        print("   • NDCG@K: 考虑排序位置的归一化折损累计增益")
        print("   • 区分度: 正样本与负样本相似度的差值（越大越好）")
        print("   • AUC: ROC曲线下面积，反映分类能力（越接近1越好）")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断评估")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("👋 程序结束")
    sys.exit(0)


if __name__ == "__main__":
    main()
