import logging
import subprocess
from typing import Annotated
from langchain_core.tools import tool
from .decorators import log_io

# 初始化日志记录器
logger = logging.getLogger(__name__)


@tool
@log_io
def bash_tool(
    cmd: Annotated[str, "The bash command to be executed."],
):
    """
    用于执行 bash 命令并完成必要操作的工具函数。

    本工具可以在 LangChain 框架中被用作一个可调用的"工具"，允许语言模型执行
    系统级别的 bash 命令，并获取执行结果。

    参数:
        cmd: 要执行的 bash 命令字符串

    返回:
        命令执行的标准输出结果或错误信息
    """
    logger.info(f"执行 Bash 命令: {cmd}")
    try:
        # 执行命令并捕获输出
        # shell=True: 允许执行 shell 命令
        # check=True: 如果命令返回非零退出状态，则抛出 CalledProcessError 异常
        # text=True: 将输出作为字符串而非字节流返回
        # capture_output=True: 捕获标准输出和标准错误
        result = subprocess.run(
            cmd, shell=True, check=True, text=True, capture_output=True
        )
        # 返回标准输出作为结果
        return result.stdout
    except subprocess.CalledProcessError as e:
        # 如果命令执行失败，返回错误信息
        # returncode: 命令的退出状态码
        # stdout: 命令的标准输出
        # stderr: 命令的标准错误
        error_message = f"命令执行失败，退出码 {e.returncode}。\n标准输出: {e.stdout}\n标准错误: {e.stderr}"
        logger.error(error_message)
        return error_message
    except Exception as e:
        # 捕获任何其他异常
        error_message = f"执行命令时发生错误: {str(e)}"
        logger.error(error_message)
        return error_message


if __name__ == "__main__":
    # 当脚本直接运行时，执行示例命令
    print(bash_tool.invoke("ls -all"))
