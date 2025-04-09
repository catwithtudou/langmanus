from readabilipy import simple_json_from_html_string  # 导入readabilipy库中的HTML解析函数

from .article import Article  # 导入自定义的Article类，用于存储提取后的文章内容


class ReadabilityExtractor:
    """
    网页可读性提取器
    用于从HTML内容中提取文章的核心内容（标题和正文），
    通过readabilipy库实现，该库底层使用Mozilla的Readability算法。

    该类的主要作用是将复杂的HTML页面简化，仅保留对用户有价值的文章内容部分，
    去除导航栏、侧边栏、广告等干扰元素。

    技术背景:
    - readabilipy是一个Python库，是Mozilla的Readability.js算法的Python封装
    - Readability算法最初由Arc90实验室开发，后被Mozilla收购并用于Firefox的阅读模式
    - 该算法通过分析DOM结构、文本密度和HTML标签特征来识别页面的主要内容
    - 常用于内容抽取、RSS生成、文章归档等场景

    设计考虑:
    - 采用单一职责原则，该类只负责内容提取，不涉及HTML获取和后续处理
    - 将提取结果封装为Article对象，便于后续处理（如转换为Markdown、存储等）
    - 未对extractor做缓存或池化处理，每次调用都会重新执行提取算法

    潜在优化:
    - 可添加提取配置参数，允许自定义提取行为（如是否保留图片、表格等）
    - 可实现错误处理机制，处理解析失败的情况
    - 可增加预处理步骤，如处理特定网站的自定义结构
    - 可添加后处理逻辑，如清理多余空白、规范化图片路径等
    """

    def extract_article(self, html: str) -> Article:
        """
        从HTML字符串中提取文章内容

        Args:
            html: 包含文章内容的HTML字符串

        Returns:
            Article: 包含提取出的标题和HTML内容的Article对象

        处理流程:
            1. 使用readabilipy库的simple_json_from_html_string函数解析HTML
            2. 从解析结果中提取标题和内容
            3. 将提取的信息封装到Article对象中返回

        注意:
            - 设置use_readability=True参数表示使用Mozilla的Readability算法进行内容提取
            - 如果原始HTML不包含有效的文章内容，可能返回空标题或内容
            - simple_json_from_html_string返回的字典还包含其他信息，如byline(作者)、
              excerpt(摘要)等，可根据需要扩展Article类来存储这些信息

        性能考虑:
            - 内容提取是CPU密集型操作，处理大型HTML文档可能较慢
            - 对于高并发场景，考虑使用异步处理或工作队列
            - HTML解析和DOM操作是主要的性能瓶颈
        """
        # 使用readabilipy解析HTML，返回包含文章信息的字典
        article = simple_json_from_html_string(html, use_readability=True)

        # 创建并返回Article实例，将提取出的标题和内容传入
        return Article(
            title=article.get("title"),  # 提取文章标题，使用get避免键不存在时出错
            html_content=article.get("content"),  # 提取文章HTML内容
        )
