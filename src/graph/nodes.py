import logging
import json
from copy import deepcopy
from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import END

from src.agents import research_agent, coder_agent, browser_agent
from src.agents.llm import get_llm_by_type
from src.config import TEAM_MEMBERS
from src.config.agents import AGENT_LLM_MAP
from src.prompts.template import apply_prompt_template
from src.tools.search import tavily_tool
from .types import State, Router

logger = logging.getLogger(__name__)

# 定义代理响应的格式模板，包含代理名称和响应内容
RESPONSE_FORMAT = "Response from {}:\n\n<response>\n{}\n</response>\n\n*Please execute the next step.*"


def research_node(state: State) -> Command[Literal["supervisor"]]:
    """
    研究代理节点，负责执行研究任务。

    该节点使用 research_agent 处理当前状态，并将其结果格式化后传递给 supervisor。
    研究代理主要负责收集和分析信息，为后续决策提供支持。

    参数:
        state: 当前工作流状态

    返回:
        返回一个Command对象，更新消息并跳转到supervisor节点
    """
    logger.info("Research agent starting task")
    result = research_agent.invoke(state)
    logger.info("Research agent completed task")
    logger.debug(f"Research agent response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format(
                        "researcher", result["messages"][-1].content
                    ),
                    name="researcher",
                )
            ]
        },
        goto="supervisor",
    )


def code_node(state: State) -> Command[Literal["supervisor"]]:
    """
    代码代理节点，负责执行Python代码相关任务。

    该节点调用coder_agent处理当前状态，执行代码编写、优化或分析等任务，
    然后将结果格式化并返回给supervisor进行下一步决策。

    参数:
        state: 当前工作流状态

    返回:
        返回一个Command对象，更新消息并跳转到supervisor节点
    """
    logger.info("Code agent starting task")
    result = coder_agent.invoke(state)
    logger.info("Code agent completed task")
    logger.debug(f"Code agent response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format(
                        "coder", result["messages"][-1].content
                    ),
                    name="coder",
                )
            ]
        },
        goto="supervisor",
    )


def browser_node(state: State) -> Command[Literal["supervisor"]]:
    """
    浏览器代理节点，负责执行Web浏览和信息获取任务。

    该节点使用browser_agent访问和处理Web内容，例如搜索结果、网页内容分析等，
    完成后将结果格式化并传递给supervisor进行下一步决策。

    参数:
        state: 当前工作流状态

    返回:
        返回一个Command对象，更新消息并跳转到supervisor节点
    """
    logger.info("Browser agent starting task")
    result = browser_agent.invoke(state)
    logger.info("Browser agent completed task")
    logger.debug(f"Browser agent response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format(
                        "browser", result["messages"][-1].content
                    ),
                    name="browser",
                )
            ]
        },
        goto="supervisor",
    )


def supervisor_node(state: State) -> Command[Literal[*TEAM_MEMBERS, "__end__"]]:
    """
    监督节点，负责协调整个工作流并决定下一步应该由哪个代理执行。

    该节点是工作流程的核心控制器，它分析当前状态，根据预定义的逻辑决定：
    - 将任务分配给哪个专门的代理（研究者、编码者、浏览器等）
    - 是否应该结束整个工作流程

    参数:
        state: 当前工作流状态

    返回:
        返回一个Command对象，指定下一个执行节点（可能是任何团队成员或结束标记）
    """
    logger.info("Supervisor evaluating next action")
    messages = apply_prompt_template("supervisor", state)
    response = (
        get_llm_by_type(AGENT_LLM_MAP["supervisor"])
        .with_structured_output(Router)
        .invoke(messages)
    )
    goto = response["next"]
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"Supervisor response: {response}")

    if goto == "FINISH":
        goto = "__end__"
        logger.info("Workflow completed")
    else:
        logger.info(f"Supervisor delegating to: {goto}")

    return Command(goto=goto, update={"next": goto})


def planner_node(state: State) -> Command[Literal["supervisor", "__end__"]]:
    """
    规划节点，负责生成整体执行计划。

    该节点在工作流开始时被调用，生成一个完整的执行计划：
    - 支持深度思考模式（使用高级推理能力的LLM）
    - 可选择在规划前进行搜索以获取更多上下文信息
    - 尝试生成JSON格式的计划，如果失败则结束工作流

    参数:
        state: 当前工作流状态

    返回:
        返回一个Command对象，包含计划内容并跳转到supervisor或结束工作流
    """
    logger.info("Planner generating full plan")
    messages = apply_prompt_template("planner", state)
    # 是否启用深度思考模式
    llm = get_llm_by_type("basic")
    if state.get("deep_thinking_mode"):
        llm = get_llm_by_type("reasoning")
    if state.get("search_before_planning"):
        searched_content = tavily_tool.invoke({"query": state["messages"][-1].content})
        messages = deepcopy(messages)
        messages[
            -1
        ].content += f"\n\n# Relative Search Results\n\n{json.dumps([{'titile': elem['title'], 'content': elem['content']} for elem in searched_content], ensure_ascii=False)}"
    stream = llm.stream(messages)
    full_response = ""
    for chunk in stream:
        full_response += chunk.content
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"Planner response: {full_response}")

    if full_response.startswith("```json"):
        full_response = full_response.removeprefix("```json")

    if full_response.endswith("```"):
        full_response = full_response.removesuffix("```")

    goto = "supervisor"
    try:
        json.loads(full_response)
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        goto = "__end__"

    return Command(
        update={
            "messages": [HumanMessage(content=full_response, name="planner")],
            "full_plan": full_response,
        },
        goto=goto,
    )


def coordinator_node(state: State) -> Command[Literal["planner", "__end__"]]:
    """
    协调员节点，负责与客户进行沟通交流。

    该节点作为用户和系统之间的接口：
    - 处理用户输入并提供适当的响应
    - 决定是否需要将请求传递给规划节点进行进一步处理
    - 如果检测到特定的"handoff_to_planner"指令，则将控制权转交给planner节点

    参数:
        state: 当前工作流状态

    返回:
        返回一个Command对象，指定下一个执行节点（planner或结束工作流）
    """
    logger.info("Coordinator talking.")
    messages = apply_prompt_template("coordinator", state)
    response = get_llm_by_type(AGENT_LLM_MAP["coordinator"]).invoke(messages)
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"reporter response: {response}")

    goto = "__end__"
    if "handoff_to_planner" in response.content:
        goto = "planner"

    return Command(
        goto=goto,
    )


def reporter_node(state: State) -> Command[Literal["supervisor"]]:
    """
    报告员节点，负责生成最终报告。

    该节点在工作流接近结束时被调用，整合之前所有代理的工作结果：
    - 汇总研究发现、代码实现和其他信息
    - 生成结构化的最终报告
    - 结果将被格式化并传递给supervisor进行最后处理

    参数:
        state: 当前工作流状态

    返回:
        返回一个Command对象，更新消息并跳转到supervisor节点
    """
    logger.info("Reporter write final report")
    messages = apply_prompt_template("reporter", state)
    response = get_llm_by_type(AGENT_LLM_MAP["reporter"]).invoke(messages)
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"reporter response: {response}")

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format("reporter", response.content),
                    name="reporter",
                )
            ]
        },
        goto="supervisor",
    )
