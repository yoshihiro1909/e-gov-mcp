#!/usr/bin/env python3
"""
基本的な法令検索の例

このスクリプトは、e-Gov法令MCPサーバーを使用して
基本的な法令検索を実行する方法を示しています。
"""

import asyncio
import json
from fastmcp import Client


async def main():
    """基本的な法令検索を実行"""
    print("=== e-Gov法令MCP 基本検索例 ===\n")
    
    # MCPサーバーに接続
    client = Client([
        "uv", "run", "python", "src/mcp_server.py"
    ])
    
    async with client:
        print("1. 民法関連の法令を検索...")
        
        # 民法関連の法令を検索
        result = await client.call_tool("search_laws", {
            "law_title": "民法",
            "limit": 3
        })
        
        # 結果を解析
        try:
            data = json.loads(result[0].text)
            print(f"検索結果: {data.get('total_count', 0)}件")
            print()
            
            # 法令の詳細を表示
            laws = data.get('laws', [])
            for i, law in enumerate(laws, 1):
                law_info = law.get('law_info', {})
                revision_info = law.get('revision_info', {})
                
                title = revision_info.get('law_title', '不明')
                law_num = law_info.get('law_num', '不明')
                promulgation_date = law_info.get('promulgation_date', '不明')
                
                print(f"{i}. {title}")
                print(f"   法令番号: {law_num}")
                print(f"   公布日: {promulgation_date}")
                print()
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"結果の解析に失敗: {e}")
            print("生の結果:")
            print(result[0].text[:500] + "..." if len(result[0].text) > 500 else result[0].text)


if __name__ == "__main__":
    asyncio.run(main())