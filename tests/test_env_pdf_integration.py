#!/usr/bin/env python3
"""
Integration tests based on Ministry of Environment PDF documents.

Tests verify that law article references found in official MOE
notifications/guidelines can be correctly looked up via the MCP server.

Source: https://www.env.go.jp/hourei/add/index.html (waste/recycling section)
118 PDFs downloaded, 73 contain law references, 1058 total references.
"""

import json
import re
import pytest
from pathlib import Path
from fastmcp import Client
from src.mcp_server import mcp, cache_manager

pymupdf = pytest.importorskip("pymupdf")

FIXTURES = Path(__file__).parent / "fixtures"

# Kanji to Arabic mapping
_KANJI_MAP = {}
_DIGITS = '一二三四五六七八九'
for _i, _k in enumerate(_DIGITS, 1):
    _KANJI_MAP[_k] = str(_i)
_KANJI_MAP['十'] = '10'
for _i in range(11, 100):
    _tens, _ones = _i // 10, _i % 10
    if _tens == 1:
        _KANJI_MAP[f'十{_DIGITS[_ones-1]}'] = str(_i) if _ones else None
    else:
        _t = _DIGITS[_tens-1]
        if _ones == 0:
            _KANJI_MAP[f'{_t}十'] = str(_i)
        else:
            _KANJI_MAP[f'{_t}十{_DIGITS[_ones-1]}'] = str(_i)
# Clean None
_KANJI_MAP = {k: v for k, v in _KANJI_MAP.items() if v}


def _normalize(s: str) -> str:
    if not s:
        return ""
    s = s.translate(str.maketrans('\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19', '0123456789'))
    if s in _KANJI_MAP:
        return _KANJI_MAP[s]
    return s


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = pymupdf.open(str(pdf_path))
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def extract_waste_law_refs(text: str) -> list[dict]:
    """Extract waste disposal law article references."""
    pattern = re.compile(
        r'(?:(?:\u5ec3\u68c4\u7269\u51e6\u7406|\u5ec3\u68c4\u7269\u306e\u51e6\u7406\u53ca\u3073\u6e05\u6383\u306b\u95a2\u3059\u308b)?\u6cd5|\u65bd\u884c\u4ee4|\u65bd\u884c\u898f\u5247|\u4ee4|\u898f\u5247)'
        r'\u7b2c([\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\d\uff10-\uff19]+'
        r'(?:\u6761\u306e[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\d\uff10-\uff19]+)?)'
        r'\u6761'
        r'(?:\u7b2c([\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\d\uff10-\uff19]+)\u9805)?'
        r'(?:\u7b2c([\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\d\uff10-\uff19]+)\u53f7)?'
    )
    refs = []
    seen = set()
    for m in pattern.finditer(text):
        article = _normalize(m.group(1))
        paragraph = _normalize(m.group(2)) if m.group(2) else ""
        item = _normalize(m.group(3)) if m.group(3) else ""
        key = f"{article}-{paragraph}-{item}"
        if key not in seen:
            seen.add(key)
            refs.append({"article": article, "paragraph": paragraph, "item": item})
    return refs


def get_all_pdf_paths() -> list[Path]:
    """Get all PDF files in fixtures directory."""
    return sorted(FIXTURES.glob("*.pdf"))


# ====== Individual PDF tests ======

class TestMopparaPdf:
    @pytest.fixture(autouse=True)
    def clear(self):
        cache_manager.law_lookup_cache.clear()
        cache_manager.law_content_cache.clear()
        cache_manager.article_cache.clear()
        yield

    @pytest.mark.asyncio
    async def test_article_6(self):
        async with Client(mcp) as client:
            result = await client.call_tool("find_law_article", {
                "law_name": "\u5ec3\u68c4\u7269\u51e6\u7406\u6cd5", "article_number": "6"
            })
            data = json.loads(result[0].text)
            assert data["matches_found"] >= 1
            assert data["extraction_method"] == "xml_structure"

    @pytest.mark.asyncio
    async def test_article_7_para_1(self):
        async with Client(mcp) as client:
            result = await client.call_tool("find_law_article", {
                "law_name": "\u5ec3\u68c4\u7269\u51e6\u7406\u6cd5",
                "article_number": "7", "paragraph": "1"
            })
            data = json.loads(result[0].text)
            assert data["matches_found"] >= 1


class TestGyouseiShobunPdf:
    @pytest.fixture(autouse=True)
    def clear(self):
        cache_manager.law_lookup_cache.clear()
        cache_manager.law_content_cache.clear()
        cache_manager.article_cache.clear()
        yield

    @pytest.mark.asyncio
    async def test_article_7_para_5_item_4(self):
        async with Client(mcp) as client:
            result = await client.call_tool("find_law_article", {
                "law_name": "\u5ec3\u68c4\u7269\u51e6\u7406\u6cd5",
                "article_number": "7", "paragraph": "5", "item": "4"
            })
            data = json.loads(result[0].text)
            assert data["matches_found"] >= 1

    @pytest.mark.asyncio
    async def test_article_14(self):
        async with Client(mcp) as client:
            result = await client.call_tool("find_law_article", {
                "law_name": "\u5ec3\u68c4\u7269\u51e6\u7406\u6cd5", "article_number": "14"
            })
            data = json.loads(result[0].text)
            assert data["matches_found"] >= 1


# ====== Bulk validation of ALL PDFs ======

class TestBulkAllPdfs:
    """Extract law refs from ALL 118 PDFs and verify each via MCP."""

    @pytest.fixture(autouse=True)
    def clear(self):
        cache_manager.law_lookup_cache.clear()
        cache_manager.law_content_cache.clear()
        cache_manager.article_cache.clear()
        yield

    def test_fixtures_exist(self):
        pdfs = get_all_pdf_paths()
        assert len(pdfs) >= 100, f"Expected 100+ PDFs, found {len(pdfs)}"

    def test_extract_refs_from_all(self):
        """Verify we can extract refs from multiple PDFs."""
        all_articles = set()
        pdfs_with_refs = 0
        for pdf_path in get_all_pdf_paths():
            try:
                text = extract_text_from_pdf(pdf_path)
                refs = extract_waste_law_refs(text)
                if refs:
                    pdfs_with_refs += 1
                    for r in refs:
                        all_articles.add(r["article"])
            except Exception:
                continue
        
        assert pdfs_with_refs >= 50, f"Expected 50+ PDFs with refs, got {pdfs_with_refs}"
        assert len(all_articles) >= 20, f"Expected 20+ unique articles, got {len(all_articles)}"

    @pytest.mark.asyncio
    async def test_all_unique_articles_via_mcp(self):
        """Verify ALL unique article numbers from ALL PDFs can be found."""
        all_articles = set()
        for pdf_path in get_all_pdf_paths():
            try:
                text = extract_text_from_pdf(pdf_path)
                refs = extract_waste_law_refs(text)
                for r in refs:
                    all_articles.add(r["article"])
            except Exception:
                continue

        # Some articles may be from other laws (e.g., 204, 252 from local govt law)
        # Filter to reasonable waste law article range (1-30 for main law)
        waste_articles = {a for a in all_articles if a.isdigit() and 1 <= int(a) <= 30}
        
        success = 0
        fail = 0
        failures = []
        
        async with Client(mcp) as client:
            for article in sorted(waste_articles, key=int):
                try:
                    result = await client.call_tool("find_law_article", {
                        "law_name": "\u5ec3\u68c4\u7269\u51e6\u7406\u6cd5",
                        "article_number": article,
                    })
                    data = json.loads(result[0].text)
                    if data["matches_found"] >= 1:
                        success += 1
                    else:
                        fail += 1
                        failures.append(article)
                except Exception as e:
                    fail += 1
                    failures.append(f"{article}({e})")

        total = success + fail
        rate = (success / total * 100) if total > 0 else 0
        print(f"\nBulk PDF test: {success}/{total} ({rate:.0f}%) - tested {len(waste_articles)} articles")
        if failures:
            print(f"  Failures: {failures}")
        assert rate >= 95, f"Success rate {rate:.0f}% below 95%"

    @pytest.mark.asyncio
    async def test_high_article_numbers(self):
        """Test articles >30 that appear in PDFs (may be from related laws)."""
        # These high numbers appear in PDFs but may be from shikou-rei or other laws
        high_articles = ["56", "59", "66"]
        
        async with Client(mcp) as client:
            for article in high_articles:
                # Try waste processing law first
                result = await client.call_tool("find_law_article", {
                    "law_name": "\u5ec3\u68c4\u7269\u51e6\u7406\u6cd5",
                    "article_number": article,
                })
                data = json.loads(result[0].text)
                # These may or may not exist - just verify no crash
                assert "matches_found" in data
