import os
import re
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState


def get_prompt_template(prompt_name: str) -> str:
    """
    获取指定名称的提示模板内容。

    该函数从文件系统中读取提示模板文件，并对模板内容进行处理：
    1. 读取与提示名称对应的.md文件
    2. 转义所有花括号，防止与后续的格式化冲突
    3. 将特殊格式的占位符 <<VAR>> 转换为标准的 {VAR} 格式

    参数:
        prompt_name: 提示模板的名称，对应于.md文件名

    返回:
        处理后的提示模板字符串
    """
    template = open(os.path.join(os.path.dirname(__file__), f"{prompt_name}.md")).read()
    # Escape curly braces using backslash
    template = template.replace("{", "{{").replace("}", "}}")
    # Replace `<<VAR>>` with `{VAR}`
    template = re.sub(r"<<([^>>]+)>>", r"{\1}", template)
    return template


def apply_prompt_template(prompt_name: str, state: AgentState) -> list:
    """
    应用提示模板，生成格式化的提示消息列表。

    该函数将指定的提示模板与当前状态相结合：
    1. 使用get_prompt_template获取原始模板
    2. 创建PromptTemplate对象并填充变量，包括当前时间和状态中的数据
    3. 返回一个包含系统提示和状态消息的完整消息列表

    这种设计使得提示可以动态适应当前的对话上下文和状态信息。

    参数:
        prompt_name: 提示模板的名称
        state: 当前代理状态对象，包含消息历史和其他上下文信息

    返回:
        一个消息列表，包含格式化的系统提示和状态中的消息历史
    """
    system_prompt = PromptTemplate(
        input_variables=["CURRENT_TIME"],
        template=get_prompt_template(prompt_name),
    ).format(CURRENT_TIME=datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"), **state)
    return [{"role": "system", "content": system_prompt}] + state["messages"]
