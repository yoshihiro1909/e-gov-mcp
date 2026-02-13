import asyncio, json
from fastmcp import Client
from src.mcp_server import mcp

async def main():
    async with Client(mcp) as client:
        # 規則12条の7の6 全文 - structured data
        r = await client.call_tool("find_law_article", {
            "law_name": "廃棄物処理法施行規則",
            "article_number": "12_7_6"
        })
        for item in r:
            if hasattr(item, 'text'):
                data = json.loads(item.text)
                print("=== structured_articles ===")
                sa = data.get("structured_articles", [])
                print(json.dumps(sa, ensure_ascii=False, indent=2)[:3000])
                print("\n=== articles (text) ===")
                for a in data.get("articles", []):
                    print(a[:2000] if isinstance(a, str) else json.dumps(a, ensure_ascii=False)[:2000])

        # Also check 規則4条の8 - 一般廃棄物版の定義
        print("\n\n=== 規則4条の8 (一般廃棄物) ===")
        r2 = await client.call_tool("find_law_article", {
            "law_name": "廃棄物処理法施行規則",
            "article_number": "4_8"
        })
        for item in r2:
            if hasattr(item, 'text'):
                data = json.loads(item.text)
                for a in data.get("articles", []):
                    print(a[:2000] if isinstance(a, str) else json.dumps(a, ensure_ascii=False)[:2000])

        # Check 施行令7条14号 structured
        print("\n\n=== 施行令7条14号 structured ===")
        r3 = await client.call_tool("find_law_article", {
            "law_name": "廃棄物処理法施行令",
            "article_number": "7",
            "item": "14"
        })
        for item in r3:
            if hasattr(item, 'text'):
                data = json.loads(item.text)
                sa = data.get("structured_articles", [])
                if sa:
                    # Find item 14
                    for art in sa:
                        for p in art.get("paragraphs", []):
                            for it in p.get("items", []):
                                if str(it.get("num", "")) == "14":
                                    print(json.dumps(it, ensure_ascii=False, indent=2)[:2000])

asyncio.run(main())
