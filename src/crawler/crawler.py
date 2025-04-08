import sys

from .article import Article
from .jina_client import JinaClient
from .readability_extractor import ReadabilityExtractor


class Crawler:
    """
    网页爬虫类，负责爬取指定URL的网页内容并转换为结构化的Article对象。

    该类整合了Jina爬虫服务和自定义的可读性提取器，实现了从URL到结构化文章内容的转换过程。
    主要用于为AI模型提供干净、结构化的网页内容。
    """

    def crawl(self, url: str) -> Article:
        """
        爬取指定URL的网页内容并转换为Article对象。

        工作流程：
        1. 使用JinaClient爬取原始HTML内容
        2. 使用ReadabilityExtractor从HTML中提取结构化文章
        3. 将结果封装为Article对象返回

        参数:
            url: 需要爬取的网页URL

        返回:
            Article: 包含结构化内容的文章对象
        """
        # 为了帮助语言模型更好地理解内容，我们从HTML中提取干净的
        # 文章内容，将其转换为Markdown格式，并将其分割为文本和图像块，
        # 形成统一的、适合语言模型处理的消息格式。
        #
        # Jina虽然在可读性方面不是最佳爬虫，但它使用简单且免费。
        #
        # 我们不使用Jina自带的Markdown转换器，而是使用
        # 自己的解决方案，以获得更好的可读性结果。

        # 创建Jina客户端实例
        jina_client = JinaClient()
        # 调用Jina客户端的crawl方法爬取指定URL的HTML内容
        html = jina_client.crawl(url, return_format="html")
        # 创建可读性提取器实例
        extractor = ReadabilityExtractor()
        # 使用提取器从HTML中提取结构化文章内容
        article = extractor.extract_article(html)
        # 设置文章对象的URL属性
        article.url = url
        # 返回处理完成的文章对象
        return article


if __name__ == "__main__":
    # 当脚本直接运行时的测试入口
    if len(sys.argv) == 2:
        # 如果提供了命令行参数，使用第一个参数作为URL
        url = sys.argv[1]
    else:
        # 否则使用默认URL进行测试
        url = "https://fintel.io/zh-hant/s/br/nvdc34"
    # 创建爬虫实例
    crawler = Crawler()
    # 爬取指定URL的内容
    article = crawler.crawl(url)
    # 打印转换后的Markdown格式内容
    print(article.to_markdown())
