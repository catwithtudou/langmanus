from langgraph.graph import StateGraph, START

from .types import State
from .nodes import (
    supervisor_node,
    research_node,
    code_node,
    coordinator_node,
    browser_node,
    reporter_node,
    planner_node,
)


def build_graph():
    """
    构建并返回智能体工作流图。

    这个函数定义了整个 LangManus 系统中各个智能体之间的协作关系和执行流程。
    工作流图使用 LangGraph 的 StateGraph 实现，它定义了智能体节点之间的转换逻辑。

    工作流程:
    1. 从 START 开始，首先进入 coordinator（协调员）节点
    2. coordinator 与用户交互并决定是否启动完整工作流
    3. planner 创建详细执行计划
    4. supervisor 监督整个过程并决定接下来调用哪个专家智能体
    5. 各专家智能体（researcher、coder、browser、reporter）执行特定任务
    6. 根据 supervisor 的决策，工作流在不同节点之间转换，直到任务完成

    返回:
    一个已编译的 StateGraph 对象，可以通过 invoke 方法执行
    """
    # 创建 StateGraph 构建器，使用 State 类型作为状态类型
    builder = StateGraph(State)

    # 添加从起始点到协调员节点的边，工作流总是从协调员开始
    builder.add_edge(START, "coordinator")

    # 添加各个智能体节点
    # coordinator: 协调员，负责与用户交互并理解任务
    builder.add_node("coordinator", coordinator_node)

    # planner: 规划员，负责制定详细的执行计划
    builder.add_node("planner", planner_node)

    # supervisor: 主管，负责监督执行并决定调用哪个专家智能体
    builder.add_node("supervisor", supervisor_node)

    # researcher: 研究员，负责收集和分析信息（如网络搜索）
    builder.add_node("researcher", research_node)

    # coder: 程序员，负责编写和执行代码
    builder.add_node("coder", code_node)

    # browser: 浏览器，负责网页交互和信息提取
    builder.add_node("browser", browser_node)

    # reporter: 汇报员，负责生成最终报告
    builder.add_node("reporter", reporter_node)

    # 编译工作流图并返回可执行的图对象
    return builder.compile()
