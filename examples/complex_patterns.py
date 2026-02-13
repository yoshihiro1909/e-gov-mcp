#!/usr/bin/env python3
"""
複雑パターンの条文検索例

このスクリプトは、e-Gov法令MCPサーバーv2のUltra Smart機能を使用して
「条の2」系、項・号指定などの複雑なパターンを検索する方法を示しています。
"""

import asyncio
import json
from fastmcp import Client

async def search_complex_patterns():
    """複雑なパターンの条文検索を実行"""
    print("=== 複雑パターン検索例 ===\n")
    
    client = Client([
        "uv", "run", "python", "src/mcp_server.py"
    ])
    
    async with client:
        # 複雑なパターンの例
        patterns = [
            ("会社法", "325条の3"),      # 条の2系
            ("憲法", "第9条第2項"),       # 項指定
            ("民法", "第192条"),          # 第○条形式
        ]
        
        for law_name, article in patterns:
            print(f"{law_name} {article} を検索中...")
            
            result = await client.call_tool("find_law_article", {
                "law_name": law_name,
                "article_number": article
            })
            
            try:
                data = json.loads(result[0].text)
                matches = data.get("matches_found", 0)
                print(f"  結果: {matches}件のマッチ")
                
                # 見つかった条文の一部を表示
                articles = data.get("articles", [])
                if articles:
                    preview = articles[0][:100] + "..." if len(articles[0]) > 100 else articles[0]
                    print(f"  プレビュー: {preview}")
                    
            except json.JSONDecodeError:
                print(f"  エラー: {result[0].text}")
            
            print()

if __name__ == "__main__":
    asyncio.run(search_complex_patterns())