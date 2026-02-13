import asyncio, json, sys, os, io, base64
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from mcp_server import smart_law_lookup, get_http_client

async def main():
    law_num = await smart_law_lookup("一般廃棄物の最終処分場及び産業廃棄物の最終処分場に係る技術上の基準を定める省令")
    async with await get_http_client() as client:
        resp = await client.get(f"/law_data/{law_num}", params={"law_full_text_format": "xml"})
        data = json.loads(resp.text)
    
    lft = data.get("law_full_text", "")
    xml_str = base64.b64decode(lft).decode("utf-8") if isinstance(lft, str) else ""
    
    # Find AppendedTable
    idx = xml_str.find("<AppdxTable")
    if idx >= 0:
        print(f"Found <AppdxTable at {idx}")
        # Get 5000 chars from there
        print(xml_str[idx:idx+5000])

asyncio.run(main())
