# 浏览器工具模块
# 该模块提供了一个用于控制和操作Web浏览器的工具类
# 支持同步和异步操作模式，集成了LangChain框架

import asyncio

from pydantic import BaseModel, Field
from typing import Optional, ClassVar, Type
from langchain.tools import BaseTool
from browser_use import AgentHistoryList, Browser, BrowserConfig
from browser_use import Agent as BrowserAgent
from src.agents.llm import vl_llm
from src.tools.decorators import create_logged_tool
from src.config import CHROME_INSTANCE_PATH

# 初始化预期的浏览器实例为空
expected_browser = None

# 如果配置中指定了Chrome实例路径，则创建一个浏览器实例
if CHROME_INSTANCE_PATH:
    expected_browser = Browser(
        config=BrowserConfig(chrome_instance_path=CHROME_INSTANCE_PATH)
    )


class BrowserUseInput(BaseModel):
    """浏览器工具的输入模型类

    继承自Pydantic的BaseModel，用于验证和规范化输入数据
    """

    # 指令字段，用于接收用户想要执行的浏览器操作的自然语言描述
    instruction: str = Field(..., description="用于浏览器操作的指令描述")


class BrowserTool(BaseTool):
    """浏览器操作工具类

    提供了一个高级接口来执行浏览器相关的操作，支持同步和异步执行模式
    继承自LangChain的BaseTool类
    """

    # 工具名称，用于在工具链中标识该工具
    name: ClassVar[str] = "browser"
    # 指定输入参数的schema
    args_schema: Type[BaseModel] = BrowserUseInput
    # 工具的详细描述，用于指导用户如何使用该工具
    description: ClassVar[str] = (
        "用于与Web浏览器交互的工具。输入应该是描述您想要执行的浏览器操作的自然语言，"
        "例如'访问google.com并搜索browser-use'，或'导航到Reddit并查找关于AI的热门帖子'。"
    )

    # 浏览器代理实例，用于执行具体的浏览器操作
    _agent: Optional[BrowserAgent] = None

    def _run(self, instruction: str) -> str:
        """同步执行浏览器任务

        Args:
            instruction: 要执行的浏览器操作的自然语言描述

        Returns:
            str: 操作结果或错误信息

        Note:
            使用事件循环来执行异步操作，确保在同步上下文中正确处理
        """
        # 创建新的浏览器代理实例
        self._agent = BrowserAgent(
            task=instruction,  # 设置任务指令
            llm=vl_llm,        # 设置语言模型
            browser=expected_browser,  # 设置浏览器实例
        )
        try:
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # 在事件循环中执行异步操作
                result = loop.run_until_complete(self._agent.run())
                # 处理返回结果：如果是AgentHistoryList类型，返回其final_result属性
                return (
                    str(result)
                    if not isinstance(result, AgentHistoryList)
                    else result.final_result
                )
            finally:
                # 确保事件循环被正确关闭
                loop.close()
        except Exception as e:
            # 捕获并返回任何执行过程中的错误
            return f"执行浏览器任务时出错: {str(e)}"

    async def _arun(self, instruction: str) -> str:
        """异步执行浏览器任务

        Args:
            instruction: 要执行的浏览器操作的自然语言描述

        Returns:
            str: 操作结果或错误信息
        """
        # 创建新的浏览器代理实例
        self._agent = BrowserAgent(
            task=instruction,
            llm=vl_llm
        )
        try:
            # 异步执行任务
            result = await self._agent.run()
            # 处理返回结果
            return (
                str(result)
                if not isinstance(result, AgentHistoryList)
                else result.final_result
            )
        except Exception as e:
            # 捕获并返回任何执行过程中的错误
            return f"执行浏览器任务时出错: {str(e)}"


# 使用装饰器创建带日志功能的工具类
BrowserTool = create_logged_tool(BrowserTool)
# 创建工具类的实例
browser_tool = BrowserTool()
