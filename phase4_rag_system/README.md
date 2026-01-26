#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段四：RAG系统集成
目标：构建完整的RAG系统
"""

"""
学习目标：
1. 设计高效的Prompt模板
2. 实现上下文动态选择
3. 优化生成参数
4. 支持多轮对话
5. 添加答案溯源功能

测试标准：
- 端到端响应准确率>80%
- 答案相关性>85%
- 多轮对话连贯性良好
- 能溯源答案来源
- 响应时间<3秒

关键技能：
- Prompt工程
- 上下文构建
- LLM集成
- 对话管理
- 答案溯源
- 错误处理
"""


# ==================== 1. Prompt工程 ====================
"""
Prompt设计原则：
1. 清晰性：指令明确，无歧义
2. 相关性：与任务高度相关
3. 简洁性：避免冗余信息
4. 上下文：提供必要的背景信息
5. 示例：提供Few-shot示例

RAG Prompt结构：
1. System Prompt：定义角色和行为
2. Context：检索到的相关文档
3. Question：用户问题
4. Instructions：生成指令
5. Examples：示例（可选）

基础Prompt模板：
```
You are a helpful assistant. Answer the question based on the following context.

Context:
{context}

Question: {question}

Answer:
```

进阶Prompt优化：
1. 添加指令
```
Based on the provided context, answer the user's question.
If the context doesn't contain the answer, say "I don't have enough information to answer this question."
Provide a concise and accurate answer.

Context:
{context}

Question: {question}

Answer:
```

2. 添加示例（Few-shot）
```
Based on the provided context, answer the user's question.
If the context doesn't contain enough information, state that clearly.

Example:
Context: RAG is Retrieval-Augmented Generation.
Question: What is RAG?
Answer: RAG is Retrieval-Augmented Generation.

Context:
{context}

Question: {question}

Answer:
```

3. 添加溯源指令
```
Answer the question based on the context.
Include citations for each piece of information used.
Format citations as [Source: document_id].

Context:
{context_with_ids}

Question: {question}

Answer:
```

Prompt优化技巧：
- 使用清晰的分隔符
- 控制上下文长度
- 明确输出格式
- 添加约束条件
- 使用结构化输出

评估指标：
- 答案准确性
- 答案相关性
- 答案完整性
- 答案简洁性
- 遵循指令程度
"""


# ==================== 2. 上下文构建策略 ====================
"""
上下文构建考虑因素：
1. 文档数量
2. 文档长度
3. 相关性分数
4. 信息密度
5. 多样性

构建策略：

策略1：简单Top-K
- 选择相似度最高的K个文档
- 优点：简单快速
- 缺点：可能遗漏相关信息

策略2：阈值过滤
- 设置最小相似度阈值
- 只保留高质量文档
- 优点：提高精确度
- 缺点：可能文档太少

策略3：平衡选择
- 固定数量的高质量文档
- 结合分数和数量
- 优点：平衡效果
- 缺点：需要调参

策略4：动态选择
- 根据查询复杂度调整
- 简单查询：少文档
- 复杂查询：多文档
- 优点：适应性强
- 缺点：复杂度高

策略5：信息去重
- 去除重复信息
- 合并相似内容
- 优点：节省空间
- 缺点：可能丢失细节

上下文模板：
```
[Source 1]
{doc1_content}

[Source 2]
{doc2_content}

...
```

优化建议：
- 使用文档ID便于溯源
- 添加文档来源信息
- 保持文档顺序（按相关性）
- 控制总长度（token限制）
- 添加分隔符提高可读性
"""


# ==================== 3. 生成参数优化 ====================
"""
关键生成参数：

1. Temperature（温度）
   - 范围：0.0-2.0
   - 低值（0.0-0.3）：更确定、保守
   - 中值（0.4-0.7）：平衡创造性和准确性
   - 高值（0.8-2.0）：更多样、创造性
   - RAG推荐：0.0-0.3（基于事实）

2. Top P（nucleus sampling）
   - 范围：0.0-1.0
   - 低值：更保守
   - 高值：更多样
   - RAG推荐：0.9-1.0

3. Max Tokens
   - 最大生成长度
   - 根据问题复杂度调整
   - 短问题：100-200 tokens
   - 长问题：300-500 tokens
   - 建议：适中设置

4. Frequency Penalty
   - 范围：-2.0-2.0
   - 避免重复内容
   - RAG推荐：0.0-0.5

5. Presence Penalty
   - 范围：-2.0-2.0
   - 鼓励新话题
   - RAG推荐：0.0

参数组合示例：
```
# 事实性问答
temperature=0.0
max_tokens=200
top_p=1.0

# 解释性回答
temperature=0.3
max_tokens=400
top_p=0.9

# 创造性应用
temperature=0.5
max_tokens=500
top_p=0.9
```

优化方法：
1. A/B测试
2. 人工评估
3. 自动化评估
4. 用户反馈
5. 持续调优
"""


# ==================== 4. 多轮对话管理 ====================
"""
对话管理挑战：
1. 上下文累积
2. 话题切换
3. 信息遗忘
4. 对话长度限制
5. 用户意图变化

管理策略：

策略1：滑动窗口
- 只保留最近N轮对话
- 优点：简单
- 缺点：可能丢失重要信息

策略2：重要信息提取
- 提取关键实体和意图
- 保留重要信息
- 优点：信息密度高
- 缺点：需要提取器

策略3：摘要机制
- 对历史对话进行摘要
- 保留摘要+最近几轮
- 优点：平衡
- 缺点：可能丢失细节

策略4：分层管理
- 当前轮：详细记录
- 近期轮：压缩存储
- 远期轮：摘要存储
- 优点：灵活
- 缺点：复杂

对话状态跟踪：
- 用户意图
- 当前话题
- 已知信息
- 待解决问题
- 检索历史

实现步骤：
1. 接收新问题
2. 检测话题是否切换
3. 更新对话上下文
4. 基于历史检索
5. 生成答案
6. 更新对话状态

对话模板：
```
Conversation History:
{history}

Current Question: {question}

Answer:
```

优化建议：
- 明确话题切换信号
- 使用对话状态管理器
- 限制对话历史长度
- 定期清理无用信息
"""


# ==================== 5. 答案溯源 ====================
"""
溯源的重要性：
- 提高可信度
- 便于验证
- 增强透明度
- 支持进一步探索

溯源方法：

方法1：文档ID标注
- 每个文档分配唯一ID
- 在答案中引用文档ID
- 格式：[Doc ID: xxx]

方法2：段落标注
- 标注具体段落或句子
- 更精确的溯源
- 格式：[Doc ID: xxx, Para: y]

方法3：高亮显示
- 高亮答案中的关键信息
- 显示来源文档
- 可视化展示

方法4：完整引用
- 提供完整的源文本
- 最详尽
- 但可能冗余

实现步骤：
1. 在上下文中添加文档ID
2. 生成时要求引用来源
3. 解析引用信息
4. 格式化展示

Prompt示例：
```
Answer the question based on the context.
Include citations for each piece of information.
Format citations as [Source: document_id].

Context:
[Source 1]
RAG is Retrieval-Augmented Generation technology.

[Source 2]
RAG combines retrieval and generation.

Question: What is RAG?
Answer:
```

输出示例：
```
RAG (Retrieval-Augmented Generation) is a technology that combines
retrieval and generation [Source 1][Source 2].
```

溯源展示：
1. 文本形式
2. 链接形式
3. 弹窗形式
4. 高亮形式

优化建议：
- 清晰的引用格式
- 可点击的源链接
- 显示源文档片段
- 支持多源引用
"""


# ==================== 6. 错误处理 ====================
"""
常见错误类型：
1. 检索失败
2. LLM API错误
3. 超时
4. 格式错误
5. 内容不安全

错误处理策略：

策略1：降级处理
- 检索失败→使用通用知识
- LLM失败→使用规则回复
- 超时→返回缓存结果

策略2：重试机制
- 临时错误：自动重试
- 指数退避：避免服务器压力
- 最大重试次数：3次

策略3：用户反馈
- 明确告知错误
- 提供解决方案
- 记录错误日志

策略4：监控告警
- 错误率监控
- 性能监控
- 及时告警

实现示例：
```python
try:
    # 检索
    context = retrieve(query)
except Exception as e:
    logger.error(f"检索失败: {e}")
    context = None

try:
    # 生成
    answer = generate(context, question)
except Exception as e:
    logger.error(f"生成失败: {e}")
    answer = "抱歉，我现在无法回答您的问题。请稍后再试。"

return answer
```

优化建议：
- 优雅降级
- 详细日志
- 用户友好
- 及时告警
"""


# ==================== 7. 实践任务清单 ====================
"""
基础任务：
□ 4.1 设计基础Prompt模板
□ 4.2 优化Prompt指令
□ 4.3 添加Few-shot示例
□ 4.4 实现上下文构建
□ 4.5 实现文档选择策略
□ 4.6 实现上下文压缩
□ 4.7 优化生成参数
□ 4.8 实现基础RAG流程
□ 4.9 集成LLM API
□ 4.10 实现对话历史管理
□ 4.11 实现多轮对话
□ 4.12 实现答案溯源
□ 4.13 添加错误处理
□ 4.14 实现日志记录

进阶任务：
□ 4.15 实现对话状态管理
□ 4.16 实现话题切换检测
□ 4.17 实现答案评估
□ 4.18 实现缓存机制
□ 4.19 实现性能监控
□ 4.20 实现A/B测试

测试验证：
□ 4.21 端到端准确率>80%
□ 4.22 答案相关性>85%
□ 4.23 响应时间<3秒
□ 4.24 多轮对话流畅
□ 4.25 溯源功能正常

参考资料：
- Prompt Engineering Guide: https://www.promptingguide.ai/
- LangChain Chain: https://python.langchain.com/docs/modules/chains/
- OpenAI API: https://platform.openai.com/docs/api-reference/chat
- Conversation Memory: https://python.langchain.com/docs/modules/memory/
"""

print("阶段四：RAG系统集成")
print("=" * 60)
print("\n学习目标：")
print("1. 设计高效Prompt模板")
print("2. 实现上下文动态选择")
print("3. 优化生成参数")
print("4. 支持多轮对话")
print("5. 添加答案溯源")

print("\n关键技术：")
print("- Prompt工程")
print("- 上下文构建")
print("- LLM集成")
print("- 对话管理")
print("- 答案溯源")
print("- 错误处理")

print("\n系统架构：")
print("1. 用户查询")
print("2. 检索相关文档")
print("3. 构建上下文")
print("4. 生成答案")
print("5. 返回结果+溯源")
