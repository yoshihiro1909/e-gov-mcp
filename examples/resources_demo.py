#!/usr/bin/env python3
"""
リソースアクセスの例

このスクリプトは、e-Gov法令MCPサーバーが提供する
リソース（API情報、法令種別など）にアクセスする方法を示しています。
"""

import asyncio
import json
from fastmcp import Client


async def main():
    """リソースアクセスを実行"""
    print("=== e-Gov法令MCP リソースアクセス例 ===\n")
    
    # MCPサーバーに接続
    client = Client([
        "uv", "run", "python", "src/mcp_server.py"
    ])
    
    async with client:
        # 利用可能なリソース一覧を取得
        print("1. 利用可能なリソース一覧:")
        resources = await client.list_resources()
        
        for resource in resources:
            print(f"  - {resource.uri}")
            if resource.description:
                print(f"    説明: {resource.description}")
        print()
        
        # API情報リソースを取得
        print("2. API情報:")
        try:
            api_info = await client.read_resource("api://info")
            api_data = json.loads(api_info.contents[0].text)
            
            print(f"  名前: {api_data.get('name', '不明')}")
            print(f"  説明: {api_data.get('description', '不明')}")
            print(f"  バージョン: {api_data.get('version', '不明')}")
            print(f"  ベースURL: {api_data.get('base_url', '不明')}")
            
            features = api_data.get('features', [])
            if features:
                print("  機能:")
                for feature in features:
                    print(f"    - {feature}")
                    
            supported_formats = api_data.get('supported_formats', [])
            if supported_formats:
                print(f"  対応形式: {', '.join(supported_formats)}")
                
            print()
            
        except Exception as e:
            print(f"  エラー: API情報の取得に失敗 - {e}")
            print()
        
        # 法令種別リソースを取得
        print("3. 法令種別一覧:")
        try:
            law_types = await client.read_resource("schema://law_types")
            law_types_data = json.loads(law_types.contents[0].text)
            
            types_dict = law_types_data.get('law_types', {})
            print("  利用可能な法令種別:")
            for code, description in types_dict.items():
                print(f"    {code}: {description}")
                
            usage = law_types_data.get('usage', '')
            if usage:
                print(f"  使用方法: {usage}")
                
            note = law_types_data.get('note', '')
            if note:
                print(f"  注意: {note}")
                
            print()
            
        except Exception as e:
            print(f"  エラー: 法令種別情報の取得に失敗 - {e}")
            print()
        
        # 利用可能なツール一覧を取得
        print("4. 利用可能なツール一覧:")
        tools = await client.list_tools()
        
        for tool in tools:
            print(f"  - {tool.name}")
            if tool.description:
                # 説明の最初の行のみ表示
                desc_lines = tool.description.split('\n')
                first_line = desc_lines[0].strip()
                if first_line:
                    print(f"    説明: {first_line}")
        
        print()
        print("詳細な使用例については、他のサンプルスクリプトを参照してください。")


if __name__ == "__main__":
    asyncio.run(main())