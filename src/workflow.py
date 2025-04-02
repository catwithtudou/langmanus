import logging
from src.config import TEAM_MEMBERS
from src.graph import build_graph

# 配置日志系统
# 设置默认日志级别为 INFO，并定义日志输出格式
logging.basicConfig(
    level=logging.INFO,  # 默认日志级别为 INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def enable_debug_logging():
    """启用 DEBUG 级别的日志记录，提供更详细的执行信息。"""
    logging.getLogger("src").setLevel(logging.DEBUG)


# 创建当前模块的日志记录器
logger = logging.getLogger(__name__)

# 创建智能体工作流图
# 调用 build_graph() 函数构建整个智能体协作的工作流图
# 这个图定义了智能体之间的协作关系和执行流程
graph = build_graph()


def run_agent_workflow(user_input: str, debug: bool = False):
    """运行智能体工作流处理用户输入的查询。

    该函数是整个 LangManus 系统的核心入口，它接收用户查询，
    初始化工作流状态，并执行由多个智能体组成的协作流程。

    参数:
        user_input: 用户的查询或请求内容
        debug: 如果为 True，启用 DEBUG 级别的日志记录，提供更详细的执行信息

    返回:
        包含工作流完成后的最终状态的字典，包括所有消息历史等信息
    """
    # 验证输入不能为空
    if not user_input:
        raise ValueError("Input could not be empty")

    # 如果启用了调试模式，设置更详细的日志级别
    if debug:
        enable_debug_logging()

    logger.info(f"Starting workflow with user input: {user_input}")

    # 调用工作流图的 invoke 方法，传入初始状态
    result = graph.invoke(
        {
            # 常量
            "TEAM_MEMBERS": TEAM_MEMBERS,  # 团队成员列表，定义在 src/config/__init__.py

            # 运行时变量
            "messages": [{"role": "user", "content": user_input}],  # 初始化消息历史，包含用户输入
            "deep_thinking_mode": True,  # 启用深度思考模式，使用更强大的推理模型
            "search_before_planning": True,  # 在规划前进行搜索，获取相关信息
        }
    )

    # 记录最终工作流状态（仅在 DEBUG 模式下输出，因为可能很长）
    logger.debug(f"Final workflow state: {result}")
    logger.info("Workflow completed successfully")
    return result


# 如果直接运行此脚本，将会输出工作流图的 Mermaid 图表表示
# 这有助于可视化智能体之间的协作关系
if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())
