from langgraph.prebuilt import create_react_agent

from src.prompts import apply_prompt_template
from src.tools import (
    bash_tool,
    browser_tool,
    crawl_tool,
    python_repl_tool,
    tavily_tool,
)

from .llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP

# 定义三种不同功能的智能代理

# 1. 研究型代理 (Research Agent)
# 该代理主要负责信息检索和网络爬取工作
# 使用 tavily_tool 进行搜索和 crawl_tool 进行网页爬取
research_agent = create_react_agent(
    # 根据配置文件中指定的LLM类型获取对应的语言模型
    get_llm_by_type(AGENT_LLM_MAP["researcher"]),
    # 为研究代理分配专用工具集：搜索和爬虫工具
    tools=[tavily_tool, crawl_tool],
    # 使用动态提示模板，根据当前状态生成适合研究任务的提示
    prompt=lambda state: apply_prompt_template("researcher", state),
)

# 2. 代码开发代理 (Coder Agent)
# 该代理主要负责编写和执行代码
# 使用 python_repl_tool 执行Python代码和 bash_tool 执行系统命令
coder_agent = create_react_agent(
    # 根据配置文件指定的LLM类型获取对应的代码开发专用语言模型
    get_llm_by_type(AGENT_LLM_MAP["coder"]),
    # 为代码开发代理分配专用工具集：Python解释器和Bash命令执行工具
    tools=[python_repl_tool, bash_tool],
    # 使用动态提示模板，根据当前状态生成适合编程任务的提示
    prompt=lambda state: apply_prompt_template("coder", state),
)

# 3. 浏览器代理 (Browser Agent)
# 该代理主要负责与网页交互，如点击、填表等操作
# 使用 browser_tool 进行网页浏览和交互
browser_agent = create_react_agent(
    # 根据配置文件指定的LLM类型获取对应的浏览器操作专用语言模型
    get_llm_by_type(AGENT_LLM_MAP["browser"]),
    # 为浏览器代理分配专用工具：浏览器自动化工具
    tools=[browser_tool],
    # 使用动态提示模板，根据当前状态生成适合网页交互任务的提示
    prompt=lambda state: apply_prompt_template("browser", state),
)
