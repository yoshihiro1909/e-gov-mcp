#!/usr/bin/env python3
"""
法令本文取得の例

このスクリプトは、e-Gov法令MCPサーバーを使用して
特定の法令の全文を取得する方法を示しています。
"""

import asyncio
import json
from fastmcp import Client


async def main():
    """法令本文取得を実行"""
    print("=== e-Gov法令MCP 法令本文取得例 ===\n")
    
    # MCPサーバーに接続
    client = Client([
        "uv", "run", "python", "src/mcp_server.py"
    ])
    
    async with client:
        # まず民法を検索して法令番号を取得
        print("1. 民法を検索して法令番号を取得...")
        
        search_result = await client.call_tool("search_laws", {
            "law_title": "民法",
            "limit": 1
        })
        
        try:
            search_data = json.loads(search_result[0].text)
            laws = search_data.get('laws', [])
            
            if not laws:
                print("民法が見つかりませんでした。")
                return
                
            law = laws[0]
            law_info = law.get('law_info', {})
            law_num = law_info.get('law_num')
            law_title = law.get('revision_info', {}).get('law_title', '不明')
            
            print(f"見つかった法令: {law_title}")
            print(f"法令番号: {law_num}")
            print()
            
            # 法令本文を取得
            print("2. 法令本文を取得中...")
            
            content_result = await client.call_tool("get_law_content", {
                "law_num": law_num,
                "response_format": "json"
            })
            
            # 結果を解析
            content_data = json.loads(content_result[0].text)
            
            # 基本情報を表示
            law_info_full = content_data.get('law_info', {})
            revision_info = content_data.get('revision_info', {})
            
            print(f"法令名: {revision_info.get('law_title', '不明')}")
            print(f"法令番号: {law_info_full.get('law_num', '不明')}")
            print(f"公布日: {law_info_full.get('promulgation_date', '不明')}")
            print(f"法令種別: {law_info_full.get('law_type', '不明')}")
            print()
            
            # 法令本文の一部を表示
            law_full_text = content_data.get('law_full_text')
            if law_full_text:
                print("3. 法令本文（抜粋）:")
                
                if isinstance(law_full_text, dict):
                    # JSON形式の場合
                    tag = law_full_text.get('tag', '不明')
                    print(f"ルート要素: {tag}")
                    
                    # 子要素を再帰的に探索して条文を見つける
                    def find_articles(node, depth=0):
                        if depth > 3:  # 深すぎる場合は停止
                            return
                            
                        if isinstance(node, dict):
                            tag = node.get('tag', '')
                            if tag in ['Article', 'Paragraph']:
                                attr = node.get('attr', {})
                                num = attr.get('Num', '')
                                if num:
                                    indent = "  " * depth
                                    print(f"{indent}{tag} {num}:")
                                    
                                    children = node.get('children', [])
                                    for child in children:
                                        if isinstance(child, str) and child.strip():
                                            text = child.strip()
                                            # 長い場合は省略
                                            if len(text) > 200:
                                                text = text[:200] + "..."
                                            print(f"{indent}  {text}")
                                            break
                                        elif isinstance(child, dict):
                                            find_articles(child, depth + 1)
                            else:
                                children = node.get('children', [])
                                for child in children:
                                    if isinstance(child, dict):
                                        find_articles(child, depth + 1)
                    
                    find_articles(law_full_text)
                    
                elif isinstance(law_full_text, str):
                    # 文字列形式の場合
                    print(law_full_text[:1000] + "..." if len(law_full_text) > 1000 else law_full_text)
                    
            else:
                print("法令本文が見つかりませんでした。")
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"エラー: 結果の解析に失敗 - {e}")
        except Exception as e:
            print(f"エラー: 処理に失敗 - {e}")


if __name__ == "__main__":
    asyncio.run(main())