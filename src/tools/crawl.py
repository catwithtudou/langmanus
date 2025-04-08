import logging
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from .decorators import log_io

from src.crawler import Crawler

# 初始化日志记录器
logger = logging.getLogger(__name__)


@tool
@log_io
def crawl_tool(
    url: Annotated[str, "The url to crawl."],
) -> HumanMessage:
    """
    用于爬取网页内容并将其转换为可读的Markdown格式的工具函数。

    该工具作为LangChain框架中的一个可调用"工具"，允许语言模型对指定URL进行
    网页爬取，并获取结构化的内容。返回的内容会被格式化为适合AI模型阅读的格式。

    参数:
        url: 需要爬取的网页URL地址

    返回:
        HumanMessage对象: 包含爬取内容的消息对象，以便于在LangChain对话流中使用
        或者在爬取失败时返回错误信息字符串
    """
    try:
        # 创建爬虫实例
        crawler = Crawler()
        # 调用爬虫的crawl方法爬取指定URL的内容
        # 返回的article对象包含了网页的结构化内容
        article = crawler.crawl(url)
        # 将爬取的内容转换为消息格式并包装为HumanMessage对象返回
        # 这样可以直接在聊天流中使用这些内容
        return {"role": "user", "content": article.to_message()}
    except BaseException as e:
        # 捕获所有可能的异常，确保工具不会因为爬取错误而崩溃
        error_msg = f"爬取失败。错误信息: {repr(e)}"
        # 记录错误日志
        logger.error(error_msg)
        # 将错误信息返回给调用者
        return error_msg
