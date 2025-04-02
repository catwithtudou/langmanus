"""
LangManus 项目的入口脚本。
这个脚本是整个 LangManus 多智能体系统的启动点，负责接收用户输入并执行智能体工作流。
"""

from src.workflow import run_agent_workflow

if __name__ == "__main__":
    import sys

    # 获取用户查询
    # 如果通过命令行参数提供了查询，则使用命令行参数
    # 例如：python main.py 计算 DeepSeek R1 在 HuggingFace 上的影响力指数
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        # 否则通过标准输入获取用户查询
        user_query = input("Enter your query: ")

    # 执行智能体工作流
    # user_input: 用户输入的查询内容
    # debug=True: 启用调试模式，会输出更详细的日志信息
    # 返回值 result 包含完整的工作流执行结果和消息历史
    result = run_agent_workflow(user_input=user_query, debug=True)

    # 打印对话历史
    # 遍历 result["messages"] 中的所有消息，并按照角色（role）格式化输出
    # 这些消息来自不同的智能体（如协调员、规划员、研究员等）
    print("\n=== Conversation History ===")
    for message in result["messages"]:
        role = message.type  # 消息发送者的角色（如 human、ai、system 等）
        print(f"\n[{role.upper()}]: {message.content}")
