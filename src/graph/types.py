from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import MessagesState

from src.config import TEAM_MEMBERS

# 定义路由选项
# 路由选项包括所有团队成员（智能体）和完成标记（FINISH）
# 这些选项将用于 supervisor 节点决定下一步调用哪个智能体
OPTIONS = TEAM_MEMBERS + ["FINISH"]


class Router(TypedDict):
    """
    路由决策类型，用于确定下一步应该调用哪个智能体。

    如果任务已完成不需要更多工作，则路由到 "FINISH"。
    supervisor 节点会返回这个类型的对象来指导工作流程的流向。
    """

    next: Literal[*OPTIONS]  # 下一个要执行的智能体名称或 "FINISH"


class State(MessagesState):
    """
    智能体系统的状态类，继承自 MessagesState 并添加了额外字段。

    这个类定义了工作流执行过程中需要维护的状态信息，包括：
    - 继承自 MessagesState 的消息历史
    - 常量配置
    - 运行时变量

    所有节点函数都接收并可能更新这个状态对象。
    """

    # 常量配置
    TEAM_MEMBERS: list[str]  # 团队成员列表，包含所有可用的智能体名称

    # 运行时变量
    next: str  # 下一个要执行的智能体名称
    full_plan: str  # 完整执行计划（由 planner 生成的 JSON 格式计划）
    deep_thinking_mode: bool  # 是否启用深度思考模式，使用更强大的推理模型
    search_before_planning: bool  # 是否在规划前执行搜索获取信息
