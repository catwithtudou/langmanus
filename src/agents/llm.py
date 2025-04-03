from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from typing import Optional

from src.config import (
    REASONING_MODEL,
    REASONING_BASE_URL,
    REASONING_API_KEY,
    BASIC_MODEL,
    BASIC_BASE_URL,
    BASIC_API_KEY,
    VL_MODEL,
    VL_BASE_URL,
    VL_API_KEY,
)
from src.config.agents import LLMType


def create_openai_llm(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs,
) -> ChatOpenAI:
    """
    创建一个ChatOpenAI实例，使用指定的配置参数。

    该函数构建OpenAI语言模型的接口实例，支持自定义API端点和密钥。
    它处理可选参数（如base_url和api_key），只有在提供了有效值时才将它们添加到配置中。

    参数:
        model: 要使用的OpenAI模型名称
        base_url: 可选的自定义API基础URL，用于非标准端点或代理
        api_key: 可选的OpenAI API密钥
        temperature: 生成的随机性程度，0表示确定性输出
        **kwargs: 传递给ChatOpenAI构造函数的其他参数

    返回:
        配置好的ChatOpenAI实例
    """
    # 只有在base_url不为None或空字符串时才将其包含在参数中
    llm_kwargs = {"model": model, "temperature": temperature, **kwargs}

    if base_url:  # 这将处理None或空字符串
        llm_kwargs["base_url"] = base_url

    if api_key:  # 这将处理None或空字符串
        llm_kwargs["api_key"] = api_key

    return ChatOpenAI(**llm_kwargs)


def create_deepseek_llm(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs,
) -> ChatDeepSeek:
    """
    创建一个ChatDeepSeek实例，使用指定的配置参数。

    类似于create_openai_llm函数，但针对DeepSeek模型。
    注意DeepSeek使用api_base而不是base_url作为参数名称，这是针对不同API提供商的适配。

    参数:
        model: 要使用的DeepSeek模型名称
        base_url: 可选的自定义API基础URL
        api_key: 可选的DeepSeek API密钥
        temperature: 生成的随机性程度，0表示确定性输出
        **kwargs: 传递给ChatDeepSeek构造函数的其他参数

    返回:
        配置好的ChatDeepSeek实例
    """
    # 只有在base_url不为None或空字符串时才将其包含在参数中
    llm_kwargs = {"model": model, "temperature": temperature, **kwargs}

    if base_url:  # 这将处理None或空字符串
        llm_kwargs["api_base"] = base_url

    if api_key:  # 这将处理None或空字符串
        llm_kwargs["api_key"] = api_key

    return ChatDeepSeek(**llm_kwargs)


# 用于缓存LLM实例的字典
_llm_cache: dict[LLMType, ChatOpenAI | ChatDeepSeek] = {}


def get_llm_by_type(llm_type: LLMType) -> ChatOpenAI | ChatDeepSeek:
    """
    根据类型获取LLM实例。如果缓存中已有该类型的实例，则返回缓存的实例。

    这个函数实现了一个简单的缓存机制，避免重复创建相同类型的LLM实例，
    提高性能并减少资源消耗。

    根据传入的类型，函数会创建不同的LLM实例：
    - "reasoning": 使用DeepSeek模型，适用于需要更强推理能力的任务
    - "basic": 使用OpenAI模型，适用于一般的对话任务
    - "vision": 使用OpenAI的视觉模型，适用于处理图像相关任务

    参数:
        llm_type: LLM类型，定义在LLMType中的字符串字面量

    返回:
        对应类型的LLM实例（ChatOpenAI或ChatDeepSeek）

    抛出:
        ValueError: 当提供了未知的LLM类型时
    """
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]

    if llm_type == "reasoning":
        llm = create_deepseek_llm(
            model=REASONING_MODEL,
            base_url=REASONING_BASE_URL,
            api_key=REASONING_API_KEY,
        )
    elif llm_type == "basic":
        llm = create_openai_llm(
            model=BASIC_MODEL,
            base_url=BASIC_BASE_URL,
            api_key=BASIC_API_KEY,
        )
    elif llm_type == "vision":
        llm = create_openai_llm(
            model=VL_MODEL,
            base_url=VL_BASE_URL,
            api_key=VL_API_KEY,
        )
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")

    _llm_cache[llm_type] = llm
    return llm


# 初始化不同用途的LLM实例 - 这些实例会被缓存
reasoning_llm = get_llm_by_type("reasoning")  # 推理型LLM，适用于复杂推理任务
basic_llm = get_llm_by_type("basic")          # 基础型LLM，适用于一般对话
vl_llm = get_llm_by_type("vision")            # 视觉语言LLM，适用于处理图像相关任务


if __name__ == "__main__":
    # 测试代码：演示如何使用流式输出
    stream = reasoning_llm.stream("what is mcp?")
    full_response = ""
    for chunk in stream:
        full_response += chunk.content
    print(full_response)

    # 测试基础型和视觉型LLM
    basic_llm.invoke("Hello")
    vl_llm.invoke("Hello")
