"""
搜索工具模块 - 集成了 Tavily 搜索功能

该模块提供了基于 Tavily API 的网络搜索功能，并集成了日志记录功能。
Tavily 是一个 AI 优化的搜索引擎，专门用于 LLM 应用。
"""

import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from src.config import TAVILY_MAX_RESULTS
from .decorators import create_logged_tool

# 初始化日志记录器
logger = logging.getLogger(__name__)

# 使用装饰器创建带有日志功能的 Tavily 搜索工具
# TavilySearchResults 是 LangChain 提供的搜索工具，可以返回结构化的搜索结果
# create_logged_tool 是一个工厂函数，用于为工具类添加日志记录功能
LoggedTavilySearch = create_logged_tool(TavilySearchResults)

# 初始化 Tavily 搜索工具实例
# name: 工具的唯一标识符
# max_results: 限制每次搜索返回的最大结果数量
tavily_tool = LoggedTavilySearch(name="tavily_search", max_results=TAVILY_MAX_RESULTS)
