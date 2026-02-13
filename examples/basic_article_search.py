#!/usr/bin/env python3
"""
基本的な条文検索の例

このスクリプトは、e-Gov法令MCPサーバーのfind_law_articleツールを使用して
特定の条文を検索する方法を示しています。
"""

import asyncio
from fastmcp import Client

async def search_civil_code():
    """民法192条（即時取得）を検索"""
    print("=== 基本的な条文検索例 ===\n")
    
    # uv run でサーバーを起動
    client = Client([
        "uv", "run", "python", "src/mcp_server.py"
    ])
    
    async with client:
        print("民法192条（即時取得）を検索中...")
        
        # 民法192条（即時取得）を検索
        result = await client.call_tool("find_law_article", {
            "law_name": "民法",
            "article_number": "192"
        })
        
        print("検索結果:")
        print(result[0].text)

if __name__ == "__main__":
    asyncio.run(search_civil_code())