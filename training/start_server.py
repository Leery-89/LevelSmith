#!/usr/bin/env python3
"""
启动LevelSmith Web服务器
"""

import uvicorn
from api import app

if __name__ == "__main__":
    print("启动 LevelSmith Web服务器...")
    print("访问地址: http://localhost:8000")
    uvicorn.run(app, host="localhost", port=8001)