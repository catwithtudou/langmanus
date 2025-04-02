"""
LangManus API 服务器启动脚本。
该脚本负责启动 LangManus 的 HTTP API 服务，允许通过 API 调用智能体工作流。
"""

import logging
import uvicorn

# 配置日志系统
# 设置默认日志级别为 INFO，并定义日志输出格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# 创建当前模块的日志记录器
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting LangManus API server")
    # 使用 uvicorn 启动 FastAPI 应用
    uvicorn.run(
        "src.api.app:app",  # 应用程序路径，指向 src/api/app.py 中的 app 实例
        host="0.0.0.0",     # 监听所有网络接口
        port=8000,          # 监听端口 8000
        reload=True,        # 启用热重载，代码更改时自动重启服务器
        log_level="info",   # 设置 uvicorn 日志级别为 info
    )
