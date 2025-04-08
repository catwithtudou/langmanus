# Python REPL工具模块
# 该模块提供了一个Python代码执行环境（REPL: Read-Eval-Print Loop）的工具
# 用于在Agent运行过程中动态执行Python代码，进行数据分析或计算

import logging
from typing import Annotated
from langchain_core.tools import tool  # 导入LangChain工具装饰器
from langchain_experimental.utilities import PythonREPL  # 导入Python REPL实用工具
from .decorators import log_io  # 导入自定义日志装饰器

# 初始化Python REPL环境和日志记录器
repl = PythonREPL()  # 创建Python代码执行环境实例
logger = logging.getLogger(__name__)  # 获取模块级别的日志记录器


@tool  # LangChain工具装饰器，将函数注册为LangChain可用的工具
@log_io  # 自定义日志装饰器，用于记录工具的输入和输出
def python_repl_tool(
    code: Annotated[
        str, "The python code to execute to do further analysis or calculation."
    ],
):
    """使用此工具执行Python代码，进行数据分析或计算。

    如果想要查看某个值的输出，应使用`print(...)`打印出来，这样用户才能看到结果。

    Args:
        code (str): 要执行的Python代码字符串

    Returns:
        str: 代码执行结果或错误信息
    """
    logger.info("执行Python代码")  # 记录开始执行的日志
    try:
        # 使用REPL环境执行传入的Python代码
        result = repl.run(code)
        logger.info("代码执行成功")  # 记录执行成功的日志
    except BaseException as e:
        # 捕获执行过程中的任何异常
        error_msg = f"执行失败。错误: {repr(e)}"
        logger.error(error_msg)  # 记录错误日志
        return error_msg  # 返回错误信息

    # 格式化执行结果，包含原始代码和标准输出
    result_str = f"成功执行:\n```python\n{code}\n```\n输出结果: {result}"
    return result_str  # 返回格式化后的结果
