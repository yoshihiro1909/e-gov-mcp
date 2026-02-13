#!/usr/bin/env python3
"""
キーワード検索の例

このスクリプトは、e-Gov法令MCPサーバーを使用して
法令内容のキーワード検索を実行する方法を示しています。
"""

import asyncio
import json
from fastmcp import Client


async def main():
    """キーワード検索を実行"""
    print("=== e-Gov法令MCP キーワード検索例 ===\n")
    
    # MCPサーバーに接続
    client = Client([
        "uv", "run", "python", "src/mcp_server.py"
    ])
    
    async with client:
        # 複数のキーワードでテスト
        keywords = ["契約", "相続", "損害賠償"]
        
        for keyword in keywords:
            print(f"'{keyword}'で検索中...")
            
            try:
                # キーワード検索を実行
                result = await client.call_tool("search_laws_by_keyword", {
                    "keyword": keyword,
                    "law_type": "Act",  # 法律のみに限定
                    "limit": 5
                })
                
                # 結果を解析
                data = json.loads(result[0].text)
                total_count = data.get('total_count', 0)
                
                print(f"  検索結果: {total_count}件")
                
                # 最初の数件の結果を表示
                laws = data.get('laws', [])
                for i, law in enumerate(laws[:3], 1):
                    law_info = law.get('law_info', {})
                    revision_info = law.get('revision_info', {})
                    
                    title = revision_info.get('law_title', '不明')
                    law_num = law_info.get('law_num', '不明')
                    
                    print(f"    {i}. {title} ({law_num})")
                    
                    # 該当する文章があれば表示
                    sentences = law.get('sentences', [])
                    if sentences:
                        sentence = sentences[0]
                        text = sentence.get('text', '')
                        if text:
                            # 長い場合は省略
                            display_text = text[:100] + "..." if len(text) > 100 else text
                            print(f"       内容: {display_text}")
                
                print()
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  エラー: 結果の解析に失敗 - {e}")
                print()
            except Exception as e:
                print(f"  エラー: 検索に失敗 - {e}")
                print()


if __name__ == "__main__":
    asyncio.run(main())