import logging

from src.config import TEAM_MEMBERS
from src.graph import build_graph
from langchain_community.adapters.openai import convert_message_to_dict
import uuid

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 默认日志级别为INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def enable_debug_logging():
    """启用调试级别的日志记录，以获取更详细的执行信息。"""
    logging.getLogger("src").setLevel(logging.DEBUG)


logger = logging.getLogger(__name__)

# 创建工作流图
graph = build_graph()

# 协调器消息的缓存
coordinator_cache = []
MAX_CACHE_SIZE = 2  # 缓存的最大大小


async def run_agent_workflow(
    user_input_messages: list,
    debug: bool = False,
    deep_thinking_mode: bool = False,
    search_before_planning: bool = False,
):
    """运行代理工作流处理用户输入。

    这是工作流服务的主要入口点，它协调多个代理的工作流程，处理用户请求并生成响应。
    整个过程采用事件流的方式，通过异步迭代器返回各个阶段的事件。

    Args:
        user_input_messages: 用户请求消息列表
        debug: 如果为True，则启用调试级别的日志记录
        deep_thinking_mode: 如果为True，启用深度思考模式
        search_before_planning: 如果为True，在规划前执行搜索

    Returns:
        工作流完成后的最终状态

    Yields:
        工作流执行过程中的各种事件
    """
    if not user_input_messages:
        raise ValueError("Input could not be empty")

    if debug:
        enable_debug_logging()

    logger.info(f"Starting workflow with user input: {user_input_messages}")

    # 生成唯一的工作流ID
    workflow_id = str(uuid.uuid4())

    # 需要流式处理的LLM代理列表
    streaming_llm_agents = [*TEAM_MEMBERS, "planner", "coordinator"]

    # 在每个工作流开始时重置协调器缓存
    global coordinator_cache
    coordinator_cache = []
    global is_handoff_case
    is_handoff_case = False

    # 从图中异步流式获取事件
    # TODO: extract message content from object, specifically for on_chat_model_stream
    async for event in graph.astream_events(
        {
            # 常量
            "TEAM_MEMBERS": TEAM_MEMBERS,
            # 运行时变量
            "messages": user_input_messages,
            "deep_thinking_mode": deep_thinking_mode,
            "search_before_planning": search_before_planning,
        },
        version="v2",
    ):
        # 解析事件数据
        kind = event.get("event")  # 事件类型
        data = event.get("data")   # 事件数据
        name = event.get("name")   # 事件名称
        metadata = event.get("metadata")  # 元数据

        # 获取节点名称（代理名称）
        node = (
            ""
            if (metadata.get("checkpoint_ns") is None)
            else metadata.get("checkpoint_ns").split(":")[0]
        )

        # 获取LangGraph步骤信息
        langgraph_step = (
            ""
            if (metadata.get("langgraph_step") is None)
            else str(metadata["langgraph_step"])
        )

        # 获取运行ID
        run_id = "" if (event.get("run_id") is None) else str(event["run_id"])

        # 处理代理启动事件
        if kind == "on_chain_start" and name in streaming_llm_agents:
            if name == "planner":
                # 当规划器启动时，标志着整个工作流的开始
                yield {
                    "event": "start_of_workflow",
                    "data": {"workflow_id": workflow_id, "input": user_input_messages},
                }
            ydata = {
                "event": "start_of_agent",
                "data": {
                    "agent_name": name,
                    "agent_id": f"{workflow_id}_{name}_{langgraph_step}",
                },
            }
        # 处理代理结束事件
        elif kind == "on_chain_end" and name in streaming_llm_agents:
            ydata = {
                "event": "end_of_agent",
                "data": {
                    "agent_name": name,
                    "agent_id": f"{workflow_id}_{name}_{langgraph_step}",
                },
            }
        # 处理LLM开始生成事件
        elif kind == "on_chat_model_start" and node in streaming_llm_agents:
            ydata = {
                "event": "start_of_llm",
                "data": {"agent_name": node},
            }
        # 处理LLM结束生成事件
        elif kind == "on_chat_model_end" and node in streaming_llm_agents:
            ydata = {
                "event": "end_of_llm",
                "data": {"agent_name": node},
            }
        # 处理LLM流式输出事件 - 这是最关键的部分，处理模型生成的内容
        elif kind == "on_chat_model_stream" and node in streaming_llm_agents:
            content = data["chunk"].content
            if content is None or content == "":
                # 处理空内容消息
                if not data["chunk"].additional_kwargs.get("reasoning_content"):
                    # 跳过完全为空的消息
                    continue
                # 处理推理内容
                ydata = {
                    "event": "message",
                    "data": {
                        "message_id": data["chunk"].id,
                        "delta": {
                            "reasoning_content": (
                                data["chunk"].additional_kwargs["reasoning_content"]
                            )
                        },
                    },
                }
            else:
                # 检查消息是否来自协调器(coordinator)
                if node == "coordinator":
                    # 协调器消息需要特殊处理 - 使用缓存来决定是否传递
                    if len(coordinator_cache) < MAX_CACHE_SIZE:
                        coordinator_cache.append(content)
                        cached_content = "".join(coordinator_cache)
                        if cached_content.startswith("handoff"):
                            # 如果是切换处理的情况，标记并跳过
                            is_handoff_case = True
                            continue
                        if len(coordinator_cache) < MAX_CACHE_SIZE:
                            # 缓存尚未满，继续收集内容
                            continue
                        # 发送缓存的消息
                        ydata = {
                            "event": "message",
                            "data": {
                                "message_id": data["chunk"].id,
                                "delta": {"content": cached_content},
                            },
                        }
                    elif not is_handoff_case:
                        # 非切换处理情况，直接发送消息
                        ydata = {
                            "event": "message",
                            "data": {
                                "message_id": data["chunk"].id,
                                "delta": {"content": content},
                            },
                        }
                else:
                    # 其他代理的消息直接发送
                    ydata = {
                        "event": "message",
                        "data": {
                            "message_id": data["chunk"].id,
                            "delta": {"content": content},
                        },
                    }
        # 处理工具调用开始事件
        elif kind == "on_tool_start" and node in TEAM_MEMBERS:
            ydata = {
                "event": "tool_call",
                "data": {
                    "tool_call_id": f"{workflow_id}_{node}_{name}_{run_id}",
                    "tool_name": name,
                    "tool_input": data.get("input"),
                },
            }
        # 处理工具调用结束事件
        elif kind == "on_tool_end" and node in TEAM_MEMBERS:
            ydata = {
                "event": "tool_call_result",
                "data": {
                    "tool_call_id": f"{workflow_id}_{node}_{name}_{run_id}",
                    "tool_name": name,
                    "tool_result": data["output"].content if data.get("output") else "",
                },
            }
        else:
            # 跳过其他类型的事件
            continue
        # 产生事件
        yield ydata

    # 处理最终的切换情况
    if is_handoff_case:
        yield {
            "event": "end_of_workflow",
            "data": {
                "workflow_id": workflow_id,
                "messages": [
                    convert_message_to_dict(msg)
                    for msg in data["output"].get("messages", [])
                ],
            },
        }
