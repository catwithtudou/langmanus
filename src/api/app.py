"""
FastAPI application for LangManus.
LangManus 的 FastAPI 应用程序。
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
import asyncio
from typing import AsyncGenerator, Dict, List, Any

from src.graph import build_graph
from src.config import TEAM_MEMBERS
from src.service.workflow_service import run_agent_workflow

# 配置日志
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用实例
app = FastAPI(
    title="LangManus API",
    description="API for LangManus LangGraph-based agent workflow",
    version="0.1.0",
)

# 添加 CORS 中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源的请求
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有 HTTP 头
)

# 创建 LangGraph 工作流图
graph = build_graph()


class ContentItem(BaseModel):
    """
    内容项模型，用于表示不同类型的消息内容（文本、图像等）
    """
    type: str = Field(..., description="The type of content (text, image, etc.)")
    text: Optional[str] = Field(None, description="The text content if type is 'text'")
    image_url: Optional[str] = Field(
        None, description="The image URL if type is 'image'"
    )


class ChatMessage(BaseModel):
    """
    聊天消息模型，表示单条对话消息
    """
    role: str = Field(
        ..., description="The role of the message sender (user or assistant)"
    )
    content: Union[str, List[ContentItem]] = Field(
        ...,
        description="The content of the message, either a string or a list of content items",
    )


class ChatRequest(BaseModel):
    """
    聊天请求模型，包含完整的对话历史和配置选项
    """
    messages: List[ChatMessage] = Field(..., description="The conversation history")
    debug: Optional[bool] = Field(False, description="Whether to enable debug logging")
    deep_thinking_mode: Optional[bool] = Field(
        False, description="Whether to enable deep thinking mode"
    )
    search_before_planning: Optional[bool] = Field(
        False, description="Whether to search before planning"
    )


@app.post("/api/chat/stream")
async def chat_endpoint(request: ChatRequest, req: Request):
    """
    聊天流式响应端点，通过 LangGraph 调用代理工作流

    Args:
        request: 聊天请求对象，包含消息历史和配置选项
        req: FastAPI 请求对象，用于检查连接状态

    Returns:
        使用 Server-Sent Events 的流式响应
    """
    try:
        # 将 Pydantic 模型转换为字典并规范化内容格式
        messages = []
        for msg in request.messages:
            message_dict = {"role": msg.role}

            # 处理两种不同格式的内容：字符串或内容项列表
            if isinstance(msg.content, str):
                message_dict["content"] = msg.content
            else:
                # 对于列表类型的内容，转换为工作流期望的格式
                content_items = []
                for item in msg.content:
                    if item.type == "text" and item.text:
                        content_items.append({"type": "text", "text": item.text})
                    elif item.type == "image" and item.image_url:
                        content_items.append(
                            {"type": "image", "image_url": item.image_url}
                        )

                message_dict["content"] = content_items

            messages.append(message_dict)

        async def event_generator():
            """
            事件生成器，用于创建 SSE 流
            从代理工作流中异步获取事件并转发给客户端
            """
            try:
                async for event in run_agent_workflow(
                    messages,
                    request.debug,
                    request.deep_thinking_mode,
                    request.search_before_planning,
                ):
                    # 检查客户端是否仍然连接
                    if await req.is_disconnected():
                        logger.info("Client disconnected, stopping workflow")
                        break
                    yield {
                        "event": event["event"],  # 事件类型
                        "data": json.dumps(event["data"], ensure_ascii=False),  # 事件数据，确保正确处理中文
                    }
            except asyncio.CancelledError:
                logger.info("Stream processing cancelled")
                raise

        # 返回 SSE 响应
        return EventSourceResponse(
            event_generator(),
            media_type="text/event-stream",
            sep="\n",
        )
    except Exception as e:
        # 捕获并记录所有异常，返回 500 错误响应
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
