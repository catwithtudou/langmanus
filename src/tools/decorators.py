"""
工具装饰器模块 - 提供日志记录功能的装饰器实现

本模块实现了两种方式来为工具添加日志记录功能：
1. 函数装饰器 (@log_io)：用于装饰独立的工具函数
2. 类装饰器工厂 (create_logged_tool)：用于为工具类添加日志功能
"""

import logging
import functools
from typing import Any, Callable, Type, TypeVar

# 初始化日志记录器
logger = logging.getLogger(__name__)

# 定义泛型类型变量，用于类型注解
T = TypeVar("T")


def log_io(func: Callable) -> Callable:
    """
    函数装饰器：记录工具函数的输入参数和输出结果

    这个装饰器会在函数执行前后添加日志记录，用于跟踪函数的调用情况：
    - 执行前：记录函数名和输入参数
    - 执行后：记录函数的返回值

    Args:
        func: 需要被装饰的工具函数

    Returns:
        装饰后的函数，增加了日志记录功能
    """

    @functools.wraps(func)  # 保留原函数的元数据
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # 记录输入参数
        func_name = func.__name__
        params = ", ".join(
            [*(str(arg) for arg in args), *(f"{k}={v}" for k, v in kwargs.items())]
        )
        logger.debug(f"Tool {func_name} called with parameters: {params}")

        # 执行原函数
        result = func(*args, **kwargs)

        # 记录输出结果
        logger.debug(f"Tool {func_name} returned: {result}")

        return result

    return wrapper


class LoggedToolMixin:
    """
    Mixin类：为工具类添加日志记录功能

    通过继承这个Mixin类，工具类可以获得自动日志记录的能力。
    主要提供两个方法：
    1. _log_operation：记录工具操作的辅助方法
    2. _run：重写基类的执行方法，添加日志记录
    """

    def _log_operation(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        """
        辅助方法：记录工具操作的详细信息

        Args:
            method_name: 被调用的方法名
            *args: 位置参数
            **kwargs: 关键字参数
        """
        tool_name = self.__class__.__name__.replace("Logged", "")
        params = ", ".join(
            [*(str(arg) for arg in args), *(f"{k}={v}" for k, v in kwargs.items())]
        )
        logger.debug(f"Tool {tool_name}.{method_name} called with parameters: {params}")

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """
        重写工具类的执行方法，添加日志记录功能

        在执行原始_run方法前后添加日志记录：
        - 执行前：记录调用参数
        - 执行后：记录返回结果
        """
        self._log_operation("_run", *args, **kwargs)
        result = super()._run(*args, **kwargs)
        logger.debug(
            f"Tool {self.__class__.__name__.replace('Logged', '')} returned: {result}"
        )
        return result


def create_logged_tool(base_tool_class: Type[T]) -> Type[T]:
    """
    工厂函数：创建具有日志功能的工具类

    这是一个类装饰器工厂函数，用于创建一个新的工具类，该类继承自原始工具类
    并混入日志记录功能。

    使用示例：
    ```python
    LoggedMyTool = create_logged_tool(MyTool)
    tool_instance = LoggedMyTool()
    ```

    Args:
        base_tool_class: 原始工具类，需要被增强的类

    Returns:
        新的工具类，继承自LoggedToolMixin和原始工具类
    """

    class LoggedTool(LoggedToolMixin, base_tool_class):
        pass

    # 设置更具描述性的类名
    LoggedTool.__name__ = f"Logged{base_tool_class.__name__}"
    return LoggedTool
