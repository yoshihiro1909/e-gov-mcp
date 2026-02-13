#!/usr/bin/env python3
"""
CLI wrapper for e-Gov Law MCP Server.
Allows direct command-line access to Japanese law search tools
without needing an MCP client.

Usage:
  python cli.py find_law_article --law 民法 --article 192
  python cli.py search_laws --title 民法
  python cli.py search_keyword --keyword 損害賠償
  python cli.py get_law_content --law-num "明治二十九年法律第八十九号"
  python cli.py batch --pairs '[{"law":"民法","article":"192"},{"law":"憲法","article":"9"}]'
"""

import argparse
import asyncio
import json
import sys
import os
import io

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mcp_server import (
    smart_law_lookup,
    extract_text_from_xml,
    extract_articles_from_xml,
    find_article_in_xml,
    generate_search_patterns,
    get_http_client,
    LAW_ALIASES,
    BASIC_LAWS,
    cache_manager,
    prompt_loader,
    config_loader,
)
import re
import logging

logging.basicConfig(level=logging.WARNING)


async def cmd_find_law_article(law_name: str, article_number: str,
                               paragraph: str = None, item: str = None,
                               main_only: bool = True) -> dict:
    """Find a specific article in Japanese law using XML structure-based extraction."""
    original_law_input = law_name
    formal_law_name = law_name
    name_conversion_applied = False

    if law_name.strip() in LAW_ALIASES:
        formal_law_name = LAW_ALIASES[law_name.strip()]
        name_conversion_applied = True

    law_num = await smart_law_lookup(law_name)
    if not law_num:
        return {"error": f"Law '{law_name}' not found"}

    # Check caches: memory → disk → API
    cached_content = cache_manager.law_content_cache.get(law_num)
    if cached_content:
        data = cached_content
    else:
        disk_content = cache_manager.disk_cache.get(f"law_data:{law_num}")
        if disk_content:
            data = disk_content
            cache_manager.law_content_cache.put(law_num, data)
        else:
            async with await get_http_client() as client:
                response = await client.get(f"/law_data/{law_num}", params={
                    "law_full_text_format": "xml"
                })
                response.raise_for_status()
                data = json.loads(response.text)
                cache_manager.law_content_cache.put(law_num, data)
                cache_manager.disk_cache.put(f"law_data:{law_num}", data)

    law_full_text = data.get('law_full_text', {})

    # Primary: XML structure-based extraction
    xml_matches = find_article_in_xml(
        law_full_text, article_number,
        paragraph=paragraph, item=item,
        main_only=main_only
    )

    matches = []
    xml_used = False

    if xml_matches:
        xml_used = True
        for m in xml_matches:
            matches.append(m['full_text'])

    # Fallback: regex-based extraction
    if not matches:
        extracted_text = extract_text_from_xml(law_full_text)
        patterns = generate_search_patterns(article_number)

        for pattern in patterns:
            article_pattern = re.escape(pattern)
            for match in re.finditer(article_pattern, extracted_text):
                pos = match.start()
                context_before = extracted_text[max(0, pos-50):pos]
                context_after = extracted_text[pos:pos+100]
                if re.search(r'第\d+条.*第\d+条', context_before + context_after):
                    continue

                context_start = max(0, pos - 20)
                context_end = min(len(extracted_text), pos + 3000)
                context = extracted_text[context_start:context_end]

                strategies = [
                    f"{article_pattern}.*?(?=第[一二三四五六七八九十百千]+条)",
                    f"{article_pattern}[^第]*",
                    f"{article_pattern}.*?(?=（[^）]*）\\s*第)",
                    f"{article_pattern}.{{0,2000}}"
                ]
                for strategy in strategies:
                    found = re.findall(strategy, context, re.DOTALL | re.MULTILINE)
                    if found:
                        candidate = found[0].strip()
                        if len(candidate) > 50:
                            content_score = 10
                            if len(candidate) > 200: content_score += 3
                            elif len(candidate) > 100: content_score += 2
                            if '。' in candidate: content_score += 3
                            if '、' in candidate: content_score += 2
                            if re.search(r'[あ-ん]+', candidate): content_score += 3
                            if '―' in candidate: content_score -= 5
                            if candidate.count('第') > 3: content_score -= 2

                            if content_score > 10 and candidate not in matches:
                                if candidate.endswith(('。', '）', '）。', '号', '項', '条')):
                                    matches.append(candidate)
                                else:
                                    for ending in ['。', '）。', '号。', '項。']:
                                        if ending in candidate:
                                            last_pos = candidate.rfind(ending)
                                            if last_pos > len(candidate) * 0.7:
                                                matches.append(candidate[:last_pos + len(ending)])
                                                break
                                    else:
                                        if len(candidate) > 100:
                                            matches.append(candidate)
                            break

    law_info_data = data.get('law_info', {})
    actual_law_title = law_info_data.get('law_title', formal_law_name)
    law_id = law_info_data.get('law_id', '')

    # Build e-Gov URL with article anchor
    egov_url = ""
    if law_id:
        egov_url = f"https://laws.e-gov.go.jp/law/{law_id}"
        # Add article anchor (e-Gov uses #Mp-At_<num> format)
        if article_number:
            clean_num = re.sub(r'[第条]', '', str(article_number)).replace('の', '-').replace('_', '-')
            egov_url += f"#Mp-At_{clean_num}"

    result = {
        "search_law_name": original_law_input,
        "actual_law_title": actual_law_title,
        "law_number": law_num,
        "law_id": law_id,
        "egov_url": egov_url,
        "search_article": article_number,
        "matches_found": len(matches),
        "articles": matches[:3] if matches else [],
        "extraction_method": "xml_structure" if xml_used else "regex_fallback",
    }
    if paragraph:
        result["filtered_paragraph"] = paragraph
    if item:
        result["filtered_item"] = item
    if xml_used and xml_matches:
        result["structured_articles"] = xml_matches
    if name_conversion_applied:
        result["alias_conversion"] = f"{original_law_input} → {formal_law_name}"
    if not matches:
        result["suggestion"] = f"Article {article_number} not found in {law_name}. Verify the article number."
    return result


async def cmd_search_laws(law_title: str = "", law_type: str = "", law_num: str = "",
                          limit: int = 10, offset: int = 0) -> dict:
    """Search Japanese laws with filtering."""
    params = {"limit": limit, "offset": offset}
    if law_title: params["law_title"] = law_title
    if law_type: params["law_type"] = law_type
    if law_num: params["law_num"] = law_num

    async with await get_http_client() as client:
        response = await client.get("/laws", params=params)
        response.raise_for_status()
        return json.loads(response.text)


async def cmd_search_keyword(keyword: str, law_type: str = "", limit: int = 5) -> dict:
    """Full-text keyword search."""
    params = {"keyword": keyword.strip(), "limit": limit}
    if law_type: params["law_type"] = law_type

    async with await get_http_client() as client:
        response = await client.get("/keyword", params=params)
        response.raise_for_status()
        return json.loads(response.text)


async def cmd_get_law_content(law_id: str = "", law_num: str = "",
                               response_format: str = "json") -> dict:
    """Get full law content."""
    law_identifier = law_id if law_id else law_num
    if not law_identifier:
        return {"error": "Either --law-id or --law-num must be specified"}

    params = {}
    if response_format == "xml":
        params["law_full_text_format"] = "xml"

    async with await get_http_client() as client:
        response = await client.get(f"/law_data/{law_identifier}", params=params)
        response.raise_for_status()
        data = json.loads(response.text)

        response_str = json.dumps(data, ensure_ascii=False)
        if len(response_str) > 800000:
            law_info = data.get('law_info', {})
            return {
                "law_info": law_info,
                "warning": "Law text exceeds 800KB. Use find_law_article for specific articles.",
                "size_bytes": len(response_str),
            }
        return data


async def cmd_find_related_laws(law_name: str) -> dict:
    """Find related laws (施行令, 施行規則) for a given base law."""
    resolved_name = law_name.strip()
    if resolved_name in LAW_ALIASES:
        resolved_name = LAW_ALIASES[resolved_name]

    related = config_loader.related_laws.get(resolved_name, {})

    if not related:
        async with await get_http_client() as client:
            related = {}
            for suffix, law_type in [("施行令", "CabinetOrder"), ("施行規則", "MinisterialOrdinance")]:
                search_title = resolved_name + suffix
                try:
                    response = await client.get("/laws", params={
                        "law_title": search_title,
                        "law_type": law_type,
                        "limit": 3
                    })
                    response.raise_for_status()
                    data = json.loads(response.text)
                    laws = data.get("laws", [])
                    if laws:
                        law_info = laws[0].get("revision_info", {}) or laws[0].get("law_info", {})
                        related[suffix] = law_info.get("law_title", search_title)
                        related[f"{suffix}番号"] = laws[0].get("law_info", {}).get("law_num", "")
                except Exception:
                    pass

    return {
        "base_law": resolved_name,
        "alias_used": law_name.strip() if law_name.strip() != resolved_name else None,
        "related_laws": related,
    }


async def cmd_search_articles(law_name: str, keyword: str, max_results: int = 10) -> dict:
    """Search for articles containing a keyword within a specific law."""
    law_num = await smart_law_lookup(law_name)
    if not law_num:
        return {"error": f"Law '{law_name}' not found"}

    # Check caches: memory → disk → API
    cached_content = cache_manager.law_content_cache.get(law_num)
    if cached_content:
        data = cached_content
    else:
        disk_content = cache_manager.disk_cache.get(f"law_data:{law_num}")
        if disk_content:
            data = disk_content
            cache_manager.law_content_cache.put(law_num, data)
        else:
            async with await get_http_client() as client:
                response = await client.get(f"/law_data/{law_num}", params={
                    "law_full_text_format": "xml"
                })
                response.raise_for_status()
                data = json.loads(response.text)
                cache_manager.law_content_cache.put(law_num, data)
                cache_manager.disk_cache.put(f"law_data:{law_num}", data)

    law_full_text = data.get('law_full_text', {})
    all_articles = extract_articles_from_xml(law_full_text)

    matches = []
    for article in all_articles:
        full_text = article.get("full_text", "")
        if keyword in full_text:
            matching_paragraphs = []
            for para in article.get("paragraphs", []):
                if keyword in para.get("text", ""):
                    matching_paragraphs.append(para["num"])
            # Preview
            idx = full_text.find(keyword)
            start = max(0, idx - 80)
            end = min(len(full_text), idx + len(keyword) + 80)
            preview = full_text[start:end].replace("\n", " ").strip()
            matches.append({
                "article_num": article.get("num", ""),
                "caption": article.get("caption", ""),
                "matching_paragraphs": matching_paragraphs,
                "preview": ("..." if start > 0 else "") + preview + ("..." if end < len(full_text) else ""),
            })
            if len(matches) >= max_results:
                break

    law_info = data.get('law_info', {})
    return {
        "law_name": law_name,
        "actual_law_title": law_info.get('law_title', law_name),
        "keyword": keyword,
        "total_articles_searched": len(all_articles),
        "matches_found": len(matches),
        "matches": matches,
    }


async def cmd_batch(pairs_json: str) -> dict:
    """Batch find multiple articles."""
    pairs = json.loads(pairs_json)
    results = []
    for pair in pairs[:200]:
        law = pair.get("law", "")
        article = pair.get("article", "")
        if law and article:
            result = await cmd_find_law_article(law, article)
            results.append(result)
    return {"results": results, "total": len(results)}


def main():
    parser = argparse.ArgumentParser(
        description="e-Gov Law CLI - Search and retrieve Japanese laws",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python cli.py find_law_article --law 民法 --article 192
  python cli.py search_laws --title 著作権法
  python cli.py search_keyword --keyword 損害賠償
  python cli.py get_law_content --law-num "明治二十九年法律第八十九号"
  python cli.py batch --pairs '[{"law":"民法","article":"192"}]'"""
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # find_law_article
    p_find = subparsers.add_parser("find_law_article", help="Find a specific law article")
    p_find.add_argument("--law", required=True, help="Law name (e.g., 民法, 会社法)")
    p_find.add_argument("--article", required=True, help="Article number (e.g., 192, 第9条)")
    p_find.add_argument("--paragraph", default=None, help="Paragraph number (項) to filter (e.g., 5)")
    p_find.add_argument("--item", default=None, help="Item number (号) to filter (e.g., 1)")
    p_find.add_argument("--include-suppl", action="store_true", help="Include 附則 (supplementary provisions)")

    # search_laws
    p_search = subparsers.add_parser("search_laws", help="Search laws by title/type/number")
    p_search.add_argument("--title", default="", help="Law title (partial match)")
    p_search.add_argument("--type", default="", help="Law type (Act, CabinetOrder, etc.)")
    p_search.add_argument("--num", default="", help="Law number (partial match)")
    p_search.add_argument("--limit", type=int, default=10, help="Max results")
    p_search.add_argument("--offset", type=int, default=0, help="Offset for pagination")

    # search_keyword
    p_kw = subparsers.add_parser("search_keyword", help="Full-text keyword search")
    p_kw.add_argument("--keyword", required=True, help="Search keyword")
    p_kw.add_argument("--type", default="", help="Law type filter")
    p_kw.add_argument("--limit", type=int, default=5, help="Max results")

    # get_law_content
    p_content = subparsers.add_parser("get_law_content", help="Get full law text")
    p_content.add_argument("--law-id", default="", help="Law ID")
    p_content.add_argument("--law-num", default="", help="Law number")
    p_content.add_argument("--format", default="json", choices=["json", "xml"], help="Response format")

    # find_related_laws
    p_related = subparsers.add_parser("find_related_laws", help="Find related laws (施行令, 施行規則)")
    p_related.add_argument("--law", required=True, help="Base law name (e.g., 廃棄物処理法, 民法)")

    # search_articles
    p_sa = subparsers.add_parser("search_articles", help="Search for keyword within a law's articles")
    p_sa.add_argument("--law", required=True, help="Law name (e.g., 廃棄物処理法, 民法)")
    p_sa.add_argument("--keyword", required=True, help="Keyword to search within articles")
    p_sa.add_argument("--max", type=int, default=10, help="Max results")

    # follow_chain
    p_chain = subparsers.add_parser("follow_chain", help="Follow 法→施行令→施行規則 chain")
    p_chain.add_argument("--law", required=True, help="Starting law name")
    p_chain.add_argument("--article", required=True, help="Article number")
    p_chain.add_argument("--paragraph", default="", help="Paragraph number")
    p_chain.add_argument("--depth", type=int, default=3, help="Max chain depth")

    # explain
    p_explain = subparsers.add_parser("explain", help="Deep explanation: resolve ALL cross-references from one article")
    p_explain.add_argument("--law", required=True, help="Law name (e.g., 廃棄物処理法)")
    p_explain.add_argument("--article", required=True, help="Article number")
    p_explain.add_argument("--paragraph", default="", help="Paragraph number")
    p_explain.add_argument("--deep", action="store_true", help="Also expand delegated articles' item references")

    # bulk_explain
    p_bulk = subparsers.add_parser("bulk_explain", help="Explain multiple articles at once with deduplicated cross-references")
    p_bulk.add_argument("--law", required=True, help="Law name (e.g., 廃棄物処理法)")
    p_bulk.add_argument("--articles", required=True, help="Comma-separated article numbers (e.g., 14,14の2,14の3の2)")
    p_bulk.add_argument("--deep", action="store_true", help="Also expand delegated articles' item references")

    # get_tables
    p_tables = subparsers.add_parser("get_tables", help="Extract appended tables (別表) from a law")
    p_tables.add_argument("--law", required=True, help="Law name")
    p_tables.add_argument("--table", default="", help="Filter by table name (e.g., 別表第一)")

    # reverse_lookup
    p_reverse = subparsers.add_parser("reverse_lookup", help="Find which articles require a specific document/condition")
    p_reverse.add_argument("--law", required=True, help="Base law name")
    p_reverse.add_argument("--keyword", required=True, help="Requirement to search for (e.g., 住民票)")

    # batch
    p_batch = subparsers.add_parser("batch", help="Batch find multiple articles")
    p_batch.add_argument("--pairs", required=True, help='JSON array: [{"law":"民法","article":"192"}]')

    args = parser.parse_args()

    if args.command == "find_law_article":
        result = asyncio.run(cmd_find_law_article(args.law, args.article, args.paragraph, args.item, main_only=not args.include_suppl))
    elif args.command == "search_laws":
        result = asyncio.run(cmd_search_laws(args.title, args.type, args.num, args.limit, args.offset))
    elif args.command == "search_keyword":
        result = asyncio.run(cmd_search_keyword(args.keyword, args.type, args.limit))
    elif args.command == "get_law_content":
        result = asyncio.run(cmd_get_law_content(args.law_id, args.law_num, args.format))
    elif args.command == "find_related_laws":
        result = asyncio.run(cmd_find_related_laws(args.law))
    elif args.command == "search_articles":
        result = asyncio.run(cmd_search_articles(args.law, args.keyword, args.max))
    elif args.command == "explain":
        from mcp_server import explain_law_article
        fn = explain_law_article.fn if hasattr(explain_law_article, 'fn') else explain_law_article
        result = asyncio.run(fn(args.law, args.article, args.paragraph, deep=args.deep))
    elif args.command == "bulk_explain":
        from mcp_server import bulk_explain_articles
        fn = bulk_explain_articles.fn if hasattr(bulk_explain_articles, 'fn') else bulk_explain_articles
        result = asyncio.run(fn(args.law, args.articles, deep=args.deep))
    elif args.command == "get_tables":
        from mcp_server import get_law_tables
        fn = get_law_tables.fn if hasattr(get_law_tables, 'fn') else get_law_tables
        result = asyncio.run(fn(args.law, args.table))
    elif args.command == "reverse_lookup":
        from mcp_server import reverse_lookup_requirement
        fn = reverse_lookup_requirement.fn if hasattr(reverse_lookup_requirement, 'fn') else reverse_lookup_requirement
        result = asyncio.run(fn(args.law, args.keyword))
    elif args.command == "follow_chain":
        from mcp_server import follow_law_chain
        # follow_law_chain is an MCP tool, call its underlying function
        fn = follow_law_chain.fn if hasattr(follow_law_chain, 'fn') else follow_law_chain
        result = asyncio.run(fn(args.law, args.article, args.paragraph, args.depth))
    elif args.command == "batch":
        result = asyncio.run(cmd_batch(args.pairs))
    else:
        parser.print_help()
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
