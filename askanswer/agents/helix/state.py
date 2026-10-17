# Helix 子图状态定义。
# 在标准 MessagesState 之上扩展演化循环所需字段：
# - topic / interview_qa：interview 阶段输入与产出
# - seed / lineage：seed 当前快照与历代沿革
# - artifact：execute 节点产出的方案文本
# - evaluation：evaluate 节点产出的评分与 gaps
# - generation / verdict：演化代数与终局标记
from __future__ import annotations

from langgraph.graph import MessagesState


class HelixState(MessagesState):
    topic: str
    interview_qa: list[dict]
    seed: dict
    artifact: str
    evaluation: dict
    generation: int
    lineage: list[dict]
    verdict: str
