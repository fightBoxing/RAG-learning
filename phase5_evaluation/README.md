#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段五：评估与优化
目标：建立评估体系并持续优化
"""

"""
学习目标：
1. 理解检索质量评估指标
2. 掌握生成质量评估指标
3. 集成端到端评估工具
4. 实现A/B测试方法
5. 建立性能监控和优化

测试标准：
- 建立完整评估体系
- 自动化评估覆盖>80%
- 性能指标可视化
- 持续优化机制
- 评估结果可追溯

关键技能：
- 评估指标计算
- 评估工具使用
- 数据收集
- 结果分析
- 优化策略
- 监控告警
"""


# ==================== 1. 检索质量评估 ====================
"""
检索评估指标：

1. Precision@K（精确率）
   定义：Top-K结果中相关文档的比例
   公式：Precision@K = 相关文档数 / K
   范围：[0, 1]
   越高越好

2. Recall@K（召回率）
   定义：相关文档在Top-K中的覆盖率
   公式：Recall@K = 检索到的相关文档数 / 总相关文档数
   范围：[0, 1]
   越高越好

3. MRR（Mean Reciprocal Rank）
   定义：第一个相关文档排名倒数的平均值
   公式：MRR = 平均(1 / rank_of_first_relevant)
   范围：[0, 1]
   越高越好

4. NDCG（Normalized Discounted Cumulative Gain）
   定义：考虑排序质量的归一化指标
   公式：考虑排序位置和相关性等级
   范围：[0, 1]
   越高越好

5. MAP（Mean Average Precision）
   定义：平均精确率的平均值
   公式：各查询的平均精确率的平均
   范围：[0, 1]
   越高越好

评估流程：
1. 准备测试数据集
2. 标注每个查询的相关文档
3. 执行检索
4. 计算评估指标
5. 分析结果

目标值：
- Precision@5 > 0.8
- Recall@10 > 0.7
- MRR > 0.75
- NDCG@10 > 0.8
"""


# ==================== 2. 生成质量评估 ====================
"""
生成评估指标：

1. BLEU（Bilingual Evaluation Understudy）
   用途：评估机器翻译质量
   基础：n-gram重叠率
   范围：[0, 1]
   越高越好

2. ROUGE（Recall-Oriented Understudy for Gisting Evaluation）
   ROUGE-N：n-gram召回率
   ROUGE-L：最长公共子序列
   ROUGE-S：跳跃二元组
   用途：评估摘要质量
   范围：[0, 1]
   越高越好

3. BERTScore
   用途：基于BERT的语义相似度
   基础：BERT embeddings的余弦相似度
   优点：考虑语义，不依赖精确匹配
   范围：[0, 1]
   越高越好

4. METEOR（Metric for Evaluation of Translation with Explicit ORdering）
   用途：机器翻译评估
   基础：同义词匹配、词形匹配
   优点：比BLEU更灵活
   范围：[0, 1]
   越高越好

5. Faithfulness（忠实度）
   定义：答案与检索上下文的一致性
   评估：是否包含未在上下文中的信息
   范围：[0, 1]
   越高越好

6. Answer Relevancy（答案相关性）
   定义：答案与问题的相关性
   评估：问题-答案对的语义相关性
   范围：[0, 1]
   越高越好

评估方法：
- 自动化评估：使用指标计算
- 人工评估：专家打分
- 混合评估：结合两者
- 用户反馈：用户满意度

目标值：
- ROUGE-L > 0.6
- BERTScore > 0.8
- Faithfulness > 0.85
- Answer Relevancy > 0.85
"""


# ==================== 3. 端到端评估 ====================
"""
端到端评估工具：

1. RAGAS（Retrieval Augmented Generation Assessment）
   功能：自动评估RAG系统
   指标：
   - Context Precision
   - Context Recall
   - Faithfulness
   - Answer Relevancy

2. TruLens（TruLens RAG）
   功能：RAG评估和监控
   特点：可追溯、可视化
   指标：
   - Context Relevance
   - Groundedness
   - Answer Relevance

3. DeepEval
   功能：LLM应用评估
   指标：
   - Faithfulness
   - Answer Relevancy
   - Bias Detection

4. Ragatouille
   功能：专注于检索评估
   特点：轻量级、易用
   指标：
   - MRR
   - MAP
   - NDCG

评估流程：
1. 准备测试数据
2. 执行RAG系统
3. 收集结果
4. 运行评估工具
5. 分析结果
6. 优化迭代

示例：
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

result = evaluate(
    dataset=test_dataset,
    metrics=[faithfulness, answer_relevancy]
)

print(result)
```
"""


# ==================== 4. A/B测试 ====================
"""
A/B测试原理：
- 对比两个版本的效果
- 随机分配用户到不同版本
- 统计分析差异显著性

RAG系统A/B测试场景：
1. 检索策略对比
   - 版本A：纯向量检索
   - 版本B：混合检索
   - 指标：准确率、响应时间

2. Prompt模板对比
   - 版本A：基础Prompt
   - 版本B：优化Prompt
   - 指标：答案质量

3. 模型对比
   - 版本A：GPT-3.5
   - 版本B：GPT-4
   - 指标：质量、成本

测试步骤：
1. 定义假设
2. 设计实验
3. 收集数据
4. 统计分析
5. 得出结论

统计分析方法：
1. t检验（连续变量）
2. 卡方检验（分类变量）
3. 置信区间
4. p值（显著性水平 < 0.05）

示例：
```python
from scipy import stats

# 版本A的准确率
scores_a = [0.82, 0.85, 0.78, 0.90, 0.83]

# 版本B的准确率
scores_b = [0.88, 0.92, 0.85, 0.91, 0.87]

# t检验
t_stat, p_value = stats.ttest_ind(scores_a, scores_b)

print(f"t统计量: {t_stat:.4f}")
print(f"p值: {p_value:.4f}")

if p_value < 0.05:
    print("差异显著")
else:
    print("差异不显著")
```
"""


# ==================== 5. 性能监控 ====================
"""
监控指标：

1. 检索性能
   - 查询响应时间
   - 检索准确率
   - 召回率
   - MRR

2. 生成性能
   - 生成时间
   - Token使用量
   - 成本
   - 答案质量

3. 系统性能
   - QPS（每秒查询数）
   - 延迟（P50, P95, P99）
   - 错误率
   - 可用性

4. 用户体验
   - 用户满意度
   - 反馈评分
   - 使用频率
   - 留存率

监控工具：
1. Prometheus + Grafana
   - 时序数据库
   - 可视化Dashboard
   - 告警规则

2. ELK Stack
   - 日志收集
   - 分析和展示
   - 告警通知

3. 自定义监控
   - 数据库存储
   - Web界面
   - 实时更新

监控流程：
1. 数据收集
2. 数据存储
3. 数据可视化
4. 告警触发
5. 问题分析

优化策略：
1. 响应时间优化
   - 索引优化
   - 缓存
   - 并行处理

2. 准确率优化
   - 调整参数
   - 改进算法
   - 增加数据

3. 成本优化
   - 模型选择
   - 批量处理
   - 缓存机制
"""


# ==================== 6. 实践任务清单 ====================
"""
基础任务：
□ 5.1 实现Precision@K计算
□ 5.2 实现Recall@K计算
□ 5.3 实现MRR计算
□ 5.4 实现NDCG计算
□ 5.5 实现BLEU计算
□ 5.6 实现ROUGE计算
□ 5.7 实现BERTScore计算
□ 5.8 集成RAGAS评估
□ 5.9 集成TruLens评估
□ 5.10 设计评估数据集
□ 5.11 实现自动化评估流程
□ 5.12 设计A/B测试
□ 5.13 实现监控数据收集
□ 5.14 建立Dashboard

进阶任务：
□ 5.15 实现人工评估接口
□ 5.16 实现用户反馈收集
□ 5.17 实现持续评估
□ 5.18 实现告警系统
□ 5.19 实现优化建议
□ 5.20 建立优化闭环

测试验证：
□ 5.21 评估体系完整
□ 5.22 自动化评估>80%
□ 5.23 Dashboard可视化
□ 5.24 监控指标齐全
□ 5.25 持续优化机制

参考资料：
- RAGAS: https://docs.ragas.io/
- TruLens: https://www.trulens.org/
- DeepEval: https://docs.confident-ai.com/
- ROUGE: https://github.com/pltrdy/rouge
"""

print("阶段五：评估与优化")
print("=" * 60)
print("\n学习目标：")
print("1. 掌握检索质量评估")
print("2. 掌握生成质量评估")
print("3. 集成端到端评估工具")
print("4. 实现A/B测试")
print("5. 建立性能监控")

print("\n评估体系：")
print("- 检索指标：Precision, Recall, MRR, NDCG")
print("- 生成指标：BLEU, ROUGE, BERTScore")
print("- 端到端：RAGAS, TruLens")
print("- 用户指标：满意度、反馈")

print("\n优化闭环：")
print("1. 收集数据")
print("2. 评估分析")
print("3. 优化迭代")
print("4. 持续监控")
