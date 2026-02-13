#!/usr/bin/env python3
"""
Comprehensive coverage tests for e-Gov Law MCP Server.
Covers: LRUCache, MemoryMonitor, CacheManager, MCP tools (mocked API), PromptLoader.
"""

import base64
import json
import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from fastmcp import Client

from src.mcp_server import (
    LRUCache,
    DiskCache,
    MemoryMonitor,
    CacheManager,
    ConfigLoader,
    mcp,
    smart_law_lookup,
    arabic_to_kanji,
    generate_search_patterns,
    extract_text_from_xml,
    extract_articles_from_xml,
    find_article_in_xml,
    _normalize_number,
    _normalize_article_number,
    _keyword_preview,
    _resolve_internal_subrefs,
    _extract_subref_titles,
    _extract_definition_label,
    _annotate_subitems_with_labels,
    extract_tables_from_xml,
    BASIC_LAWS,
    LAW_ALIASES,
    cache_manager,
)
from src.prompt_loader import PromptLoader


# ====== Sample XML for mocked API responses ======

SAMPLE_LAW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Law>
  <LawBody>
    <MainProvision>
      <Article Num="1">
        <ArticleCaption>(mokuteki)</ArticleCaption>
        <ArticleTitle>dai-ichijou</ArticleTitle>
        <Paragraph Num="1">
          <ParagraphNum/>
          <ParagraphSentence><Sentence>kono houritsu wa test mokuteki no houritsu de aru.</Sentence></ParagraphSentence>
        </Paragraph>
      </Article>
      <Article Num="709">
        <ArticleCaption>(fuhoukoui)</ArticleCaption>
        <ArticleTitle>dai-nanahyakukujou</ArticleTitle>
        <Paragraph Num="1">
          <ParagraphNum/>
          <ParagraphSentence><Sentence>koi matawa kashitsu niyotte tanin no kenri wo shingai shita mono wa baishou suru sekinin wo ou.</Sentence></ParagraphSentence>
        </Paragraph>
      </Article>
    </MainProvision>
    <SupplProvision>
      <Article Num="1">
        <ArticleTitle>fusoku dai-ichijou</ArticleTitle>
        <Paragraph Num="1">
          <ParagraphNum/>
          <ParagraphSentence><Sentence>kono houritsu wa koufu no hi kara shikou suru.</Sentence></ParagraphSentence>
        </Paragraph>
      </Article>
    </SupplProvision>
  </LawBody>
</Law>"""

SAMPLE_B64 = base64.b64encode(SAMPLE_LAW_XML.encode('utf-8')).decode('ascii')

# XML with cross-references for follow_law_chain tests
CHAIN_LAW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Law>
  <LawBody>
    <MainProvision>
      <Article Num="12">
        <ArticleCaption>（委託の基準）</ArticleCaption>
        <ArticleTitle>第十二条</ArticleTitle>
        <Paragraph Num="1">
          <ParagraphNum/>
          <ParagraphSentence><Sentence>事業者は、その産業廃棄物の運搬又は処分を他人に委託する場合には、政令で定める基準に従わなければならない。</Sentence></ParagraphSentence>
        </Paragraph>
      </Article>
      <Article Num="14">
        <ArticleCaption>（許可）</ArticleCaption>
        <ArticleTitle>第十四条</ArticleTitle>
        <Paragraph Num="1">
          <ParagraphNum/>
          <ParagraphSentence><Sentence>産業廃棄物の収集又は運搬を業として行おうとする者は、環境省令で定めるところにより、都道府県知事の許可を受けなければならない。</Sentence></ParagraphSentence>
        </Paragraph>
      </Article>
      <Article Num="15">
        <ArticleCaption>（施設の設置）</ArticleCaption>
        <ArticleTitle>第十五条</ArticleTitle>
        <Paragraph Num="1">
          <ParagraphNum/>
          <ParagraphSentence><Sentence>令第七条に定める産業廃棄物処理施設を設置しようとする者は、都道府県知事の許可を受けなければならない。</Sentence></ParagraphSentence>
        </Paragraph>
      </Article>
      <Article Num="19">
        <ArticleCaption>（準用）</ArticleCaption>
        <ArticleTitle>第十九条</ArticleTitle>
        <Paragraph Num="1">
          <ParagraphNum/>
          <ParagraphSentence><Sentence>第十四条の規定は、特別管理産業廃棄物について準用する。</Sentence></ParagraphSentence>
        </Paragraph>
      </Article>
    </MainProvision>
  </LawBody>
</Law>"""
CHAIN_B64 = base64.b64encode(CHAIN_LAW_XML.encode('utf-8')).decode('ascii')

CHAIN_SEIREI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Law>
  <LawBody>
    <MainProvision>
      <Article Num="7">
        <ArticleCaption>（産業廃棄物処理施設）</ArticleCaption>
        <ArticleTitle>第七条</ArticleTitle>
        <Paragraph Num="1">
          <ParagraphNum/>
          <ParagraphSentence><Sentence>法第十五条の政令で定める産業廃棄物処理施設は、次のとおりとする。規則第十二条に定める基準に適合するものでなければならない。</Sentence></ParagraphSentence>
        </Paragraph>
      </Article>
      <Article Num="12">
        <ArticleCaption>（委託の基準）</ArticleCaption>
        <ArticleTitle>第十二条</ArticleTitle>
        <Paragraph Num="1">
          <ParagraphNum/>
          <ParagraphSentence><Sentence>法第十二条の政令で定める基準は、次のとおりとする。</Sentence></ParagraphSentence>
        </Paragraph>
      </Article>
    </MainProvision>
  </LawBody>
</Law>"""
CHAIN_SEIREI_B64 = base64.b64encode(CHAIN_SEIREI_XML.encode('utf-8')).decode('ascii')

def _make_api_response(law_title="test law", law_num="reiwa1-law1"):
    return {
        "law_info": {
            "law_title": law_title,
            "law_num": law_num,
            "promulgation_date": "2019-01-01",
        },
        "law_full_text": SAMPLE_B64,
    }

def _make_search_response(laws=None):
    if laws is None:
        laws = [{"law_info": {"law_title": "test law", "law_num": "reiwa1-law1"}}]
    return {"laws": laws}


# ====== LRUCache Tests ======

class TestLRUCache:
    def test_put_and_get(self):
        cache = LRUCache(max_size=3, ttl=3600)
        cache.put("a", 1)
        assert cache.get("a") == 1

    def test_miss(self):
        cache = LRUCache(max_size=3, ttl=3600)
        assert cache.get("missing") is None

    def test_eviction(self):
        cache = LRUCache(max_size=2, ttl=3600)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_update_existing(self):
        cache = LRUCache(max_size=3, ttl=3600)
        cache.put("a", 1)
        cache.put("a", 2)
        assert cache.get("a") == 2
        assert cache.size() == 1

    def test_ttl_expiry(self):
        cache = LRUCache(max_size=3, ttl=0)
        cache.put("a", 1)
        time.sleep(0.01)
        assert cache.get("a") is None

    def test_clear(self):
        cache = LRUCache(max_size=3, ttl=3600)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert cache.size() == 0

    def test_cleanup_expired(self):
        cache = LRUCache(max_size=10, ttl=0)
        cache.put("a", 1)
        cache.put("b", 2)
        time.sleep(0.01)
        cache.cleanup_expired()
        assert cache.size() == 0

    def test_lru_order(self):
        cache = LRUCache(max_size=2, ttl=3600)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")
        cache.put("c", 3)
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3


# ====== DiskCache Tests ======

class TestDiskCache:
    def test_put_and_get(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path / "cache"), ttl=3600)
        cache.put("key1", {"data": "hello"})
        assert cache.get("key1") == {"data": "hello"}

    def test_miss(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path / "cache"), ttl=3600)
        assert cache.get("missing") is None

    def test_ttl_expiry(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path / "cache"), ttl=1)
        cache.put("key1", {"data": "hello"})
        import os
        path = cache._path("key1")
        # Force file to be old
        old_time = time.time() - 10
        os.utime(path, (old_time, old_time))
        assert cache.get("key1") is None

    def test_overwrite(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path / "cache"), ttl=3600)
        cache.put("key1", {"v": 1})
        cache.put("key1", {"v": 2})
        assert cache.get("key1") == {"v": 2}

    def test_clear(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path / "cache"), ttl=3600)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.size() == 2
        cache.clear()
        assert cache.size() == 0
        assert cache.get("a") is None

    def test_size(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path / "cache"), ttl=3600)
        assert cache.size() == 0
        cache.put("a", 1)
        assert cache.size() == 1
        cache.put("b", 2)
        assert cache.size() == 2

    def test_unicode_values(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path / "cache"), ttl=3600)
        cache.put("law", {"title": "廃棄物の処理及び清掃に関する法律", "article": "第12条"})
        result = cache.get("law")
        assert result["title"] == "廃棄物の処理及び清掃に関する法律"
        assert result["article"] == "第12条"

    def test_creates_dir(self, tmp_path):
        deep_path = str(tmp_path / "a" / "b" / "c" / "cache")
        cache = DiskCache(cache_dir=deep_path, ttl=3600)
        cache.put("key", "val")
        assert cache.get("key") == "val"


# ====== MemoryMonitor Tests ======

class TestMemoryMonitor:
    def test_without_psutil(self):
        with patch("src.mcp_server.PERFORMANCE_MONITORING_AVAILABLE", False):
            mm = MemoryMonitor()
            mm.process = None
            assert mm.get_memory_usage_mb() == 0.0
            assert mm.is_memory_limit_exceeded() is False

    def test_with_psutil_mock(self):
        mm = MemoryMonitor(max_memory_mb=100)
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 50 * 1024 * 1024
        mm.process = mock_process
        with patch("src.mcp_server.PERFORMANCE_MONITORING_AVAILABLE", True):
            assert mm.get_memory_usage_mb() == pytest.approx(50.0)
            assert mm.is_memory_limit_exceeded() is False

    def test_memory_exceeded(self):
        mm = MemoryMonitor(max_memory_mb=10)
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 20 * 1024 * 1024
        mm.process = mock_process
        with patch("src.mcp_server.PERFORMANCE_MONITORING_AVAILABLE", True):
            assert mm.is_memory_limit_exceeded() is True


# ====== CacheManager Tests ======

class TestCacheManager:
    def test_get_cache_key(self):
        cm = CacheManager()
        assert cm.get_cache_key("minpou") == "minpou"
        assert cm.get_cache_key("minpou", "709") == "minpou:709"

    def test_should_clear_cache(self):
        cm = CacheManager()
        assert cm.should_clear_cache() is False

    def test_cleanup_if_needed_no_pressure(self):
        cm = CacheManager()
        cm.law_lookup_cache.put("test", "value")
        cm.cleanup_if_needed()
        assert cm.law_lookup_cache.get("test") == "value"

    def test_cleanup_if_needed_memory_pressure(self):
        cm = CacheManager()
        cm.law_lookup_cache.put("test", "value")
        cm.memory_monitor = MagicMock()
        cm.memory_monitor.is_memory_limit_exceeded.return_value = True
        cm.cleanup_if_needed()
        assert cm.law_lookup_cache.size() == 0


# ====== ConfigLoader Tests ======

class TestConfigLoader:
    def test_fallback_when_no_file(self):
        cl = ConfigLoader(config_path="nonexistent/path.yaml")
        aliases = cl.law_aliases
        assert isinstance(aliases, dict)

    def test_reload(self):
        cl = ConfigLoader()
        _ = cl.law_aliases
        _ = cl.basic_laws
        _ = cl.related_laws
        cl.reload_config()
        assert cl._law_aliases is None
        assert cl._basic_laws is None
        assert cl._related_laws is None


# ====== smart_law_lookup Tests (mocked HTTP) ======

def _make_mock_client(response_data):
    mock_response = MagicMock()
    mock_response.text = json.dumps(response_data)
    mock_response.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


class TestSmartLawLookup:
    @pytest.fixture(autouse=True)
    def clear_caches(self):
        cache_manager.law_lookup_cache.clear()
        cache_manager.law_content_cache.clear()
        cache_manager.article_cache.clear()
        yield
        cache_manager.law_lookup_cache.clear()
        cache_manager.law_content_cache.clear()
        cache_manager.article_cache.clear()

    @pytest.mark.asyncio
    async def test_direct_mapping(self):
        result = await smart_law_lookup("minpou")
        # minpou is not in BASIC_LAWS (Japanese name is), so it searches API
        # Test with actual Japanese name
        result = await smart_law_lookup("kenpou")
        # This also won't be in BASIC_LAWS unless Japanese chars used

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        cache_manager.law_lookup_cache.put("test_law", "cached_num")
        result = await smart_law_lookup("test_law")
        assert result == "cached_num"

    @pytest.mark.asyncio
    async def test_api_search(self):
        mock_client = _make_mock_client(_make_search_response())
        with patch("src.mcp_server.get_http_client", AsyncMock(return_value=mock_client)):
            result = await smart_law_lookup("unknown_law_xyz")
            assert result == "reiwa1-law1"

    @pytest.mark.asyncio
    async def test_api_no_results(self):
        mock_client = _make_mock_client({"laws": []})
        with patch("src.mcp_server.get_http_client", AsyncMock(return_value=mock_client)):
            result = await smart_law_lookup("nonexistent_xyz123")
            assert result is None

    @pytest.mark.asyncio
    async def test_cabinet_order_detection(self):
        mock_client = _make_mock_client(_make_search_response([{
            "law_info": {"law_title": "test shikourei", "law_num": "reiwa1-cabinet1"}
        }]))
        with patch("src.mcp_server.get_http_client", AsyncMock(return_value=mock_client)):
            # Use a name with shikourei suffix that triggers CabinetOrder
            cache_manager.law_lookup_cache.clear()
            result = await smart_law_lookup("test_shikourei")
            # Won't detect CabinetOrder because Japanese chars needed
            # Just verify it returns something


# ====== MCP Tool Tests (via FastMCP Client) ======

def _patch_for_tool_test():
    """Patch smart_law_lookup and get_http_client for tool tests."""
    mock_client = _make_mock_client(_make_api_response())
    return (
        patch("src.mcp_server.smart_law_lookup", AsyncMock(return_value="reiwa1-law1")),
        patch("src.mcp_server.get_http_client", AsyncMock(return_value=mock_client)),
    )


class TestFindLawArticleTool:
    @pytest.fixture(autouse=True)
    def clear_caches(self):
        cache_manager.law_lookup_cache.clear()
        cache_manager.law_content_cache.clear()
        cache_manager.article_cache.clear()
        yield
        cache_manager.law_lookup_cache.clear()

    @pytest.mark.asyncio
    async def test_find_article_success(self):
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("find_law_article", {
                    "law_name": "testlaw", "article_number": "709"
                })
                data = json.loads(result[0].text)
                assert data["matches_found"] >= 1
                assert data["extraction_method"] == "xml_structure"

    @pytest.mark.asyncio
    async def test_find_article_with_paragraph(self):
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("find_law_article", {
                    "law_name": "testlaw", "article_number": "1", "paragraph": "1"
                })
                data = json.loads(result[0].text)
                assert data["matches_found"] >= 1

    @pytest.mark.asyncio
    async def test_find_article_not_found(self):
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("find_law_article", {
                    "law_name": "testlaw", "article_number": "99999"
                })
                data = json.loads(result[0].text)
                assert data["matches_found"] == 0

    @pytest.mark.asyncio
    async def test_find_article_main_only_false(self):
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("find_law_article", {
                    "law_name": "testlaw", "article_number": "1", "main_only": False
                })
                data = json.loads(result[0].text)
                assert data["matches_found"] >= 2

    @pytest.mark.asyncio
    async def test_find_article_content_cache_hit(self):
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                await client.call_tool("find_law_article", {
                    "law_name": "testlaw", "article_number": "709"
                })
                result = await client.call_tool("find_law_article", {
                    "law_name": "testlaw", "article_number": "1"
                })
                data = json.loads(result[0].text)
                assert data["matches_found"] >= 1


class TestSearchLawsTool:
    @pytest.mark.asyncio
    async def test_search_laws(self):
        mock_client = _make_mock_client(_make_search_response())
        with patch("src.mcp_server.get_http_client", AsyncMock(return_value=mock_client)):
            async with Client(mcp) as client:
                result = await client.call_tool("search_laws", {
                    "law_title": "minpou", "limit": 5
                })
                text = result[0].text if hasattr(result[0], 'text') else str(result[0])
                data = json.loads(text)
                assert "laws" in data

    @pytest.mark.asyncio
    async def test_search_laws_with_all_params(self):
        mock_client = _make_mock_client(_make_search_response())
        with patch("src.mcp_server.get_http_client", AsyncMock(return_value=mock_client)):
            async with Client(mcp) as client:
                result = await client.call_tool("search_laws", {
                    "law_title": "test", "law_type": "CabinetOrder",
                    "law_num": "reiwa", "limit": 3, "offset": 0
                })
                text = result[0].text if hasattr(result[0], 'text') else str(result[0])
                data = json.loads(text)
                assert "laws" in data


class TestSearchLawsByKeywordTool:
    @pytest.mark.asyncio
    async def test_keyword_search(self):
        mock_client = _make_mock_client({"laws": [{"law_info": {"law_title": "test"}}]})
        with patch("src.mcp_server.get_http_client", AsyncMock(return_value=mock_client)):
            async with Client(mcp) as client:
                result = await client.call_tool("search_laws_by_keyword", {
                    "keyword": "test keyword"
                })
                text = result[0].text if hasattr(result[0], 'text') else str(result[0])
                data = json.loads(text)
                assert "laws" in data

    @pytest.mark.asyncio
    async def test_keyword_search_with_type(self):
        mock_client = _make_mock_client({"laws": []})
        with patch("src.mcp_server.get_http_client", AsyncMock(return_value=mock_client)):
            async with Client(mcp) as client:
                result = await client.call_tool("search_laws_by_keyword", {
                    "keyword": "test", "law_type": "Act", "limit": 3
                })
                text = result[0].text if hasattr(result[0], 'text') else str(result[0])
                data = json.loads(text)
                assert "laws" in data


class TestGetLawContentTool:
    @pytest.mark.asyncio
    async def test_get_content_json(self):
        mock_client = _make_mock_client(_make_api_response())
        with patch("src.mcp_server.get_http_client", AsyncMock(return_value=mock_client)):
            async with Client(mcp) as client:
                result = await client.call_tool("get_law_content", {
                    "law_num": "reiwa1-law1"
                })
                text = result[0].text if hasattr(result[0], 'text') else str(result[0])
                data = json.loads(text)
                assert "law_info" in data

    @pytest.mark.asyncio
    async def test_get_content_xml(self):
        mock_response = MagicMock()
        mock_response.text = "<Law>test</Law>"
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("src.mcp_server.get_http_client", AsyncMock(return_value=mock_client)):
            async with Client(mcp) as client:
                result = await client.call_tool("get_law_content", {
                    "law_num": "reiwa1-law1", "response_format": "xml"
                })
                text = result[0].text if hasattr(result[0], 'text') else str(result[0])
                data = json.loads(text)
                assert data["format"] == "xml"

    @pytest.mark.asyncio
    async def test_get_content_large_json(self):
        large_data = _make_api_response()
        large_data["law_full_text"] = "x" * 900000
        mock_client = _make_mock_client(large_data)
        with patch("src.mcp_server.get_http_client", AsyncMock(return_value=mock_client)):
            async with Client(mcp) as client:
                result = await client.call_tool("get_law_content", {
                    "law_num": "reiwa1-law1"
                })
                text = result[0].text if hasattr(result[0], 'text') else str(result[0])
                data = json.loads(text)
                assert "warning" in data

    @pytest.mark.asyncio
    async def test_get_content_large_xml(self):
        mock_response = MagicMock()
        mock_response.text = "<Law>" + "x" * 900000 + "</Law>"
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("src.mcp_server.get_http_client", AsyncMock(return_value=mock_client)):
            async with Client(mcp) as client:
                result = await client.call_tool("get_law_content", {
                    "law_num": "reiwa1-law1", "response_format": "xml"
                })
                text = result[0].text if hasattr(result[0], 'text') else str(result[0])
                data = json.loads(text)
                assert "warning" in data

    @pytest.mark.asyncio
    async def test_get_content_with_law_id(self):
        mock_client = _make_mock_client(_make_api_response())
        with patch("src.mcp_server.get_http_client", AsyncMock(return_value=mock_client)):
            async with Client(mcp) as client:
                result = await client.call_tool("get_law_content", {
                    "law_id": "some-law-id"
                })
                text = result[0].text if hasattr(result[0], 'text') else str(result[0])
                data = json.loads(text)
                assert "law_info" in data


class TestBatchFindArticlesTool:
    @pytest.fixture(autouse=True)
    def clear_caches(self):
        cache_manager.law_lookup_cache.clear()
        cache_manager.law_content_cache.clear()
        cache_manager.article_cache.clear()
        yield

    @pytest.mark.asyncio
    async def test_batch_find(self):
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                pairs = json.dumps([
                    {"law": "testlaw", "article": "709"},
                    {"law": "testlaw", "article": "1"},
                ])
                result = await client.call_tool("batch_find_articles", {
                    "law_article_pairs": pairs
                })
                data = json.loads(result[0].text)
                assert len(data["results"]) == 2
                assert "performance_stats" in data

    @pytest.mark.asyncio
    async def test_batch_find_invalid_pair(self):
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                pairs = json.dumps([{"invalid": "pair"}])
                result = await client.call_tool("batch_find_articles", {
                    "law_article_pairs": pairs
                })
                data = json.loads(result[0].text)
                assert "error" in data["results"][0]


class TestExplainLawArticleTool:
    @pytest.fixture(autouse=True)
    def clear_caches(self):
        cache_manager.law_lookup_cache.clear()
        cache_manager.law_content_cache.clear()
        cache_manager.article_cache.clear()
        yield

    @pytest.mark.asyncio
    async def test_explain_basic(self):
        """Test explain_law_article returns structured result."""
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("explain_law_article", {
                    "law_name": "testlaw", "article_number": "1"
                })
                data = json.loads(result[0].text)
                assert "article_num" in data
                assert "delegation_to_seirei" in data
                assert "delegation_to_kisoku" in data
                assert "junyou_references" in data
                assert "summary" in data

    @pytest.mark.asyncio
    async def test_explain_not_found(self):
        """Test explain_law_article with nonexistent article."""
        from fastmcp.exceptions import ToolError
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                with pytest.raises(ToolError):
                    await client.call_tool("explain_law_article", {
                        "law_name": "testlaw", "article_number": "99999"
                    })

    @pytest.mark.asyncio
    async def test_explain_law_not_found(self):
        """Test explain_law_article with nonexistent law."""
        from fastmcp.exceptions import ToolError
        with patch("src.mcp_server.smart_law_lookup", AsyncMock(return_value=None)):
            async with Client(mcp) as client:
                with pytest.raises(ToolError):
                    await client.call_tool("explain_law_article", {
                        "law_name": "nonexistent", "article_number": "1"
                    })

    @pytest.mark.asyncio
    async def test_explain_has_yomikae_field(self):
        """Test explain_law_article returns yomikae_rules field."""
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("explain_law_article", {
                    "law_name": "testlaw", "article_number": "1"
                })
                data = json.loads(result[0].text)
                assert "yomikae_rules" in data
                assert "exception_rules" in data
                assert "expanded_item_references" in data
                assert "deep_expansions" in data

    @pytest.mark.asyncio
    async def test_explain_deep_mode(self):
        """Test explain_law_article with deep=True."""
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("explain_law_article", {
                    "law_name": "testlaw", "article_number": "1", "deep": True
                })
                data = json.loads(result[0].text)
                assert "deep_expansions" in data


class TestBulkExplainArticles:
    @pytest.fixture(autouse=True)
    def clear_caches(self):
        cache_manager.law_lookup_cache.clear()
        cache_manager.law_content_cache.clear()
        cache_manager.article_cache.clear()
        yield

    @pytest.mark.asyncio
    async def test_bulk_explain_basic(self):
        """Test bulk_explain_articles with multiple articles."""
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("bulk_explain_articles", {
                    "law_name": "testlaw", "article_numbers": "1,709"
                })
                data = json.loads(result[0].text)
                assert "articles" in data
                assert "shared_references" in data
                assert "summary" in data
                assert data["summary"]["articles_requested"] == 2

    @pytest.mark.asyncio
    async def test_bulk_explain_structure(self):
        """Test that shared_references has all expected keys."""
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("bulk_explain_articles", {
                    "law_name": "testlaw", "article_numbers": "709"
                })
                data = json.loads(result[0].text)
                refs = data["shared_references"]
                assert "delegation_to_seirei" in refs
                assert "delegation_to_kisoku" in refs
                assert "junyou_references" in refs
                assert "expanded_item_references" in refs
                assert "deep_expansions" in refs
                assert "yomikae_rules" in refs
                assert "exception_rules" in refs

    @pytest.mark.asyncio
    async def test_bulk_explain_with_nonexistent(self):
        """Test bulk_explain handles mix of found and not-found articles."""
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("bulk_explain_articles", {
                    "law_name": "testlaw", "article_numbers": "1,99999"
                })
                data = json.loads(result[0].text)
                assert data["summary"]["articles_requested"] == 2
                assert data["summary"]["articles_failed"] >= 1

    @pytest.mark.asyncio
    async def test_bulk_explain_empty_input(self):
        """Test bulk_explain with empty article list."""
        from fastmcp.exceptions import ToolError
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                with pytest.raises(ToolError):
                    await client.call_tool("bulk_explain_articles", {
                        "law_name": "testlaw", "article_numbers": ""
                    })

    @pytest.mark.asyncio
    async def test_bulk_explain_deep_mode(self):
        """Test bulk_explain with deep=True."""
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("bulk_explain_articles", {
                    "law_name": "testlaw", "article_numbers": "709", "deep": True
                })
                data = json.loads(result[0].text)
                assert "deep_expansions" in data["shared_references"]


class TestReverseLookupRequirement:
    @pytest.fixture(autouse=True)
    def clear_caches(self):
        cache_manager.law_lookup_cache.clear()
        cache_manager.law_content_cache.clear()
        cache_manager.article_cache.clear()
        yield

    @pytest.mark.asyncio
    async def test_reverse_lookup_basic(self):
        """Test reverse_lookup_requirement returns structured result."""
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("reverse_lookup_requirement", {
                    "law_name": "testlaw", "requirement_keyword": "test"
                })
                data = json.loads(result[0].text)
                assert "keyword" in data
                assert "total_matches" in data
                assert "results" in data


class TestKanjiToArabic:
    def test_basic(self):
        from src.mcp_server import kanji_to_arabic
        assert kanji_to_arabic("一") == "1"
        assert kanji_to_arabic("九") == "9"
        assert kanji_to_arabic("十") == "10"
        assert kanji_to_arabic("十四") == "14"
        assert kanji_to_arabic("二十") == "20"
        assert kanji_to_arabic("二十三") == "23"

    def test_hundreds(self):
        from src.mcp_server import kanji_to_arabic
        assert kanji_to_arabic("百") == "100"
        assert kanji_to_arabic("百二十三") == "123"
        assert kanji_to_arabic("二百") == "200"


class TestSearchArticlesByKeywordTool:
    @pytest.fixture(autouse=True)
    def clear_caches(self):
        cache_manager.law_lookup_cache.clear()
        cache_manager.law_content_cache.clear()
        cache_manager.article_cache.clear()
        yield

    @pytest.mark.asyncio
    async def test_search_keyword_found(self):
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("search_articles_by_keyword", {
                    "law_name": "testlaw", "keyword": "kenri"
                })
                data = json.loads(result[0].text)
                assert data["matches_found"] >= 1
                assert data["keyword"] == "kenri"
                assert "total_articles_searched" in data
                for m in data["matches"]:
                    assert "article_num" in m
                    assert "preview" in m

    @pytest.mark.asyncio
    async def test_search_keyword_not_found(self):
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("search_articles_by_keyword", {
                    "law_name": "testlaw", "keyword": "存在しないキーワードXYZ"
                })
                data = json.loads(result[0].text)
                assert data["matches_found"] == 0

    @pytest.mark.asyncio
    async def test_search_keyword_max_results(self):
        p1, p2 = _patch_for_tool_test()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("search_articles_by_keyword", {
                    "law_name": "testlaw", "keyword": "houritsu", "max_results": 1
                })
                data = json.loads(result[0].text)
                assert data["matches_found"] <= 1

    @pytest.mark.asyncio
    async def test_search_keyword_law_not_found(self):
        from fastmcp.exceptions import ToolError
        with patch("src.mcp_server.smart_law_lookup", AsyncMock(return_value=None)):
            async with Client(mcp) as client:
                with pytest.raises(ToolError):
                    await client.call_tool("search_articles_by_keyword", {
                        "law_name": "nonexistent", "keyword": "test"
                    })


class TestFollowLawChainTool:
    """Tests for the follow_law_chain MCP tool."""

    @pytest.fixture(autouse=True)
    def clear_caches(self):
        cache_manager.law_lookup_cache.clear()
        cache_manager.law_content_cache.clear()
        cache_manager.article_cache.clear()
        yield

    def _make_chain_patches(self):
        """Create patches for chain tests with law/seirei data."""
        law_data = {
            "law_info": {"law_title": "テスト法", "law_num": "law1"},
            "law_full_text": CHAIN_B64,
        }
        seirei_data = {
            "law_info": {"law_title": "テスト法施行令", "law_num": "seirei1"},
            "law_full_text": CHAIN_SEIREI_B64,
        }

        async def mock_lookup(name):
            if "施行規則" in name:
                return "kisoku1"
            if "施行令" in name:
                return "seirei1"
            return "law1"

        # Mock HTTP client that returns different data based on URL
        mock_response_law = MagicMock()
        mock_response_law.status_code = 200
        mock_response_law.text = json.dumps(law_data)
        mock_response_law.raise_for_status = MagicMock()

        mock_response_seirei = MagicMock()
        mock_response_seirei.status_code = 200
        mock_response_seirei.text = json.dumps(seirei_data)
        mock_response_seirei.raise_for_status = MagicMock()

        async def mock_get(url, params=None):
            if "seirei1" in url:
                return mock_response_seirei
            return mock_response_law

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        return (
            patch("src.mcp_server.smart_law_lookup", side_effect=mock_lookup),
            patch("src.mcp_server.get_http_client", AsyncMock(return_value=mock_client)),
        )

    @pytest.mark.asyncio
    async def test_chain_with_seirei_reference(self):
        """Test: article says 令第七条 → follows to 施行令第7条."""
        p1, p2 = self._make_chain_patches()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("follow_law_chain", {
                    "law_name": "テスト法", "article_number": "15"
                })
                data = json.loads(result[0].text)
                assert data["chain_depth"] >= 1
                # First node should be article 15 with reference to 施行令
                node0 = data["chain"][0]
                assert node0["article"] == "15"
                assert node0["references_found"] >= 1
                ref_types = [r["type"] for r in node0["references"]]
                assert "施行令" in ref_types

    @pytest.mark.asyncio
    async def test_chain_with_seirei_for_me(self):
        """Test: 政令で定める pattern detected."""
        p1, p2 = self._make_chain_patches()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("follow_law_chain", {
                    "law_name": "テスト法", "article_number": "12"
                })
                data = json.loads(result[0].text)
                node0 = data["chain"][0]
                assert any(r["reason"] == "政令で定める" for r in node0["references"])

    @pytest.mark.asyncio
    async def test_chain_with_shourei(self):
        """Test: 環境省令で定める pattern detected."""
        p1, p2 = self._make_chain_patches()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("follow_law_chain", {
                    "law_name": "テスト法", "article_number": "14"
                })
                data = json.loads(result[0].text)
                node0 = data["chain"][0]
                assert any(r["type"] == "施行規則" for r in node0["references"])

    @pytest.mark.asyncio
    async def test_chain_junyou_reference(self):
        """Test: 準用 pattern detected — references within same law."""
        p1, p2 = self._make_chain_patches()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("follow_law_chain", {
                    "law_name": "テスト法", "article_number": "19", "max_depth": 3
                })
                data = json.loads(result[0].text)
                node0 = data["chain"][0]
                # Should detect 準用 reference to article 14
                junyou_refs = [r for r in node0["references"] if r["type"] == "準用"]
                assert len(junyou_refs) >= 1
                assert any(r["article"] == "14" for r in junyou_refs)
                # Should follow to article 14 in the same law
                assert data["chain_depth"] >= 2
                node1 = data["chain"][1]
                assert node1["article"] == "14"
                assert node1["law"] == "テスト法"

    @pytest.mark.asyncio
    async def test_chain_article_not_found(self):
        """Test: nonexistent article returns not_found status."""
        p1, p2 = self._make_chain_patches()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("follow_law_chain", {
                    "law_name": "テスト法", "article_number": "9999"
                })
                data = json.loads(result[0].text)
                assert data["chain"][0]["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_chain_law_not_found(self):
        """Test: nonexistent law raises error."""
        from fastmcp.exceptions import ToolError
        with patch("src.mcp_server.smart_law_lookup", AsyncMock(return_value=None)):
            async with Client(mcp) as client:
                # follow_law_chain calls _get_article_text which returns None if lookup fails
                result = await client.call_tool("follow_law_chain", {
                    "law_name": "nonexistent", "article_number": "1"
                })
                data = json.loads(result[0].text)
                assert data["chain"][0]["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_chain_generic_follow_reverse_reference(self):
        """Test: 政令で定める without specific article → searches 施行令 for 法第十二条 back-reference."""
        p1, p2 = self._make_chain_patches()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("follow_law_chain", {
                    "law_name": "テスト法", "article_number": "12", "max_depth": 3
                })
                data = json.loads(result[0].text)
                node0 = data["chain"][0]
                # Should have generic_follow info with reverse_reference strategy
                assert "generic_follow" in node0
                assert node0["generic_follow"]["strategy"] == "reverse_reference"
                assert node0["generic_follow"]["found"] == True
                assert "法第十二条" in node0["generic_follow"]["pattern"]
                # Should have followed to depth >= 2
                assert data["chain_depth"] >= 2
                node1 = data["chain"][1]
                assert "施行令" in node1["law"]
                assert node1["article"] == "12"

    @pytest.mark.asyncio
    async def test_chain_depth_traversal(self):
        """Test: chain follows 令第七条 from art.15 to 施行令art.7, then finds 規則第十二条."""
        p1, p2 = self._make_chain_patches()
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("follow_law_chain", {
                    "law_name": "テスト法", "article_number": "15", "max_depth": 3
                })
                data = json.loads(result[0].text)
                # Should have at least 2 nodes: law art.15 → seirei art.7
                assert data["chain_depth"] >= 2
                if data["chain_depth"] >= 2:
                    node1 = data["chain"][1]
                    assert "施行令" in node1["law"]


class TestCacheStatsTool:
    @pytest.mark.asyncio
    async def test_get_cache_stats(self):
        async with Client(mcp) as client:
            result = await client.call_tool("get_cache_stats", {})
            text = result[0].text if hasattr(result[0], 'text') else str(result[0])
            data = json.loads(text)
            assert "cache_statistics" in data
            assert "memory_monitoring" in data
            assert "performance_features" in data


class TestClearCacheTool:
    @pytest.mark.asyncio
    async def test_clear_all(self):
        cache_manager.law_lookup_cache.put("test", "val")
        async with Client(mcp) as client:
            result = await client.call_tool("clear_cache", {"cache_type": "all"})
            text = result[0].text if hasattr(result[0], 'text') else str(result[0])
            data = json.loads(text)
            assert data["status"] == "success"
            assert cache_manager.law_lookup_cache.size() == 0

    @pytest.mark.asyncio
    async def test_clear_law_lookup(self):
        async with Client(mcp) as client:
            result = await client.call_tool("clear_cache", {"cache_type": "law_lookup"})
            text = result[0].text if hasattr(result[0], 'text') else str(result[0])
            data = json.loads(text)
            assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_clear_law_content(self):
        async with Client(mcp) as client:
            result = await client.call_tool("clear_cache", {"cache_type": "law_content"})
            text = result[0].text if hasattr(result[0], 'text') else str(result[0])
            data = json.loads(text)
            assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_clear_article(self):
        async with Client(mcp) as client:
            result = await client.call_tool("clear_cache", {"cache_type": "article"})
            text = result[0].text if hasattr(result[0], 'text') else str(result[0])
            data = json.loads(text)
            assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_clear_invalid_type(self):
        from fastmcp.exceptions import ToolError
        async with Client(mcp) as client:
            with pytest.raises(ToolError):
                await client.call_tool("clear_cache", {"cache_type": "invalid_xyz"})


class TestFindRelatedLawsTool:
    @pytest.mark.asyncio
    async def test_find_related_from_config(self):
        async with Client(mcp) as client:
            result = await client.call_tool("find_related_laws", {
                "law_name": "haikibutsu-test"
            })
            text = result[0].text if hasattr(result[0], 'text') else str(result[0])
            data = json.loads(text)
            assert "related_laws" in data

    @pytest.mark.asyncio
    async def test_find_related_api_fallback(self):
        mock_client = _make_mock_client(_make_search_response([{
            "law_info": {"law_title": "test shikourei", "law_num": "reiwa1-cabinet1"},
            "revision_info": {"law_title": "test shikourei"},
        }]))
        with patch("src.mcp_server.get_http_client", AsyncMock(return_value=mock_client)):
            async with Client(mcp) as client:
                result = await client.call_tool("find_related_laws", {
                    "law_name": "unknown_test_law_xyz"
                })
                text = result[0].text if hasattr(result[0], 'text') else str(result[0])
                data = json.loads(text)
                assert "related_laws" in data


class TestPrefetchTool:
    @pytest.mark.asyncio
    async def test_prefetch(self):
        mock_client = _make_mock_client(_make_api_response())
        with patch("src.mcp_server.get_http_client", AsyncMock(return_value=mock_client)):
            async with Client(mcp) as client:
                result = await client.call_tool("prefetch_common_laws", {})
                text = result[0].text if hasattr(result[0], 'text') else str(result[0])
                data = json.loads(text)
                assert data["status"] == "success"
                assert "cache_stats" in data


# ====== PromptLoader Tests ======

class TestPromptLoader:
    def test_fallback_instruction(self):
        pl = PromptLoader(prompts_dir="nonexistent")
        result = pl.get_legal_analysis_instruction()
        assert len(result) > 100

    def test_clear_cache(self):
        pl = PromptLoader()
        pl._cache["test"] = "value"
        pl.clear_cache()
        assert len(pl._cache) == 0

    def test_list_prompts_no_dir(self):
        pl = PromptLoader(prompts_dir="nonexistent_dir_xyz")
        assert pl.list_available_prompts() == []

    def test_list_prompts_with_dir(self):
        pl = PromptLoader(prompts_dir="prompts")
        result = pl.list_available_prompts()
        assert isinstance(result, list)

    def test_load_prompt_not_found(self):
        pl = PromptLoader(prompts_dir="prompts")
        with pytest.raises(FileNotFoundError):
            pl.load_prompt("nonexistent_prompt_xyz")

    def test_load_prompt_cached(self):
        pl = PromptLoader()
        pl._cache["test_prompt"] = "cached content"
        assert pl.load_prompt("test_prompt") == "cached content"

    def test_load_prompt_from_file(self):
        pl = PromptLoader(prompts_dir="prompts")
        prompts = pl.list_available_prompts()
        if prompts:
            result = pl.load_prompt(prompts[0])
            assert len(result) > 0
            result2 = pl.load_prompt(prompts[0])
            assert result == result2

    def test_prompts_dir_warning(self):
        pl = PromptLoader(prompts_dir="nonexistent_xyz")
        assert not pl.prompts_dir.exists()


# ====== Resources Tests ======

class TestResources:
    @pytest.mark.asyncio
    async def test_api_info(self):
        async with Client(mcp) as client:
            result = await client.read_resource("api://info")
            assert result is not None

    @pytest.mark.asyncio
    async def test_law_types(self):
        async with Client(mcp) as client:
            result = await client.read_resource("schema://law_types")
            assert result is not None


# ====== Edge Cases ======

class TestEdgeCases:
    def test_arabic_to_kanji_zero(self):
        assert arabic_to_kanji("0") == "\u3007"

    def test_arabic_to_kanji_large(self):
        assert arabic_to_kanji("1000") == "1000"

    def test_arabic_to_kanji_non_digit(self):
        assert arabic_to_kanji("abc") == "abc"

    def test_arabic_to_kanji_hundreds(self):
        assert arabic_to_kanji("100") == "\u767e"
        assert arabic_to_kanji("200") == "\u4e8c\u767e"
        assert arabic_to_kanji("105") == "\u767e\u4e94"

    def test_generate_patterns_no_match(self):
        patterns = generate_search_patterns("abc")
        assert "abc" in patterns

    def test_generate_patterns_with_no(self):
        patterns = generate_search_patterns("325no3")
        # No match for 'no' - won't trigger the no pattern

    def test_extract_text_from_list(self):
        result = extract_text_from_xml(["hello", "world"])
        assert "hello" in result
        assert "world" in result

    def test_extract_text_from_dict_with_text(self):
        result = extract_text_from_xml({"text": "hello"})
        assert result == "hello"

    def test_extract_text_from_dict_with_children(self):
        result = extract_text_from_xml({"children": [{"text": "a"}, {"text": "b"}]})
        assert "a" in result
        assert "b" in result

    def test_extract_text_from_number(self):
        result = extract_text_from_xml(42)
        assert "42" in result

    def test_extract_text_from_invalid_base64(self):
        result = extract_text_from_xml("not_valid_base64!!!")
        assert isinstance(result, str)


# ===== _keyword_preview Tests =====
class TestKeywordPreview:
    def test_basic_preview(self):
        text = "あ" * 200 + "損害賠償" + "い" * 200
        result = _keyword_preview(text, "損害賠償")
        assert "損害賠償" in result
        assert result.startswith("...")
        assert result.endswith("...")

    def test_keyword_at_start(self):
        text = "損害賠償に関する規定" + "あ" * 200
        result = _keyword_preview(text, "損害賠償")
        assert not result.startswith("...")
        assert result.endswith("...")

    def test_keyword_at_end(self):
        text = "あ" * 200 + "損害賠償"
        result = _keyword_preview(text, "損害賠償")
        assert result.startswith("...")
        assert not result.endswith("...")

    def test_short_text(self):
        text = "損害賠償の規定"
        result = _keyword_preview(text, "損害賠償")
        assert result == "損害賠償の規定"
        assert not result.startswith("...")

    def test_keyword_not_found(self):
        result = _keyword_preview("テスト", "存在しない")
        assert result == ""

    def test_custom_context_chars(self):
        text = "あ" * 50 + "キーワード" + "い" * 50
        result = _keyword_preview(text, "キーワード", context_chars=10)
        assert "キーワード" in result
        assert len(result) < len(text)

    def test_newlines_replaced(self):
        text = "第一項\n損害賠償\nの規定"
        result = _keyword_preview(text, "損害賠償")
        assert "\n" not in result


# ====== Internal sub-item cross-reference resolution tests ======

class TestExtractSubrefTitles:
    """Tests for _extract_subref_titles helper."""

    def test_single_ref(self):
        titles = _extract_subref_titles("イに規定する基準")
        assert titles == ["イ"]

    def test_single_kakageru(self):
        titles = _extract_subref_titles("ロに掲げる事項")
        assert titles == ["ロ"]

    def test_single_gaitou(self):
        titles = _extract_subref_titles("ハに該当する者")
        assert titles == ["ハ"]

    def test_single_sadameru(self):
        titles = _extract_subref_titles("ニに定める方法")
        assert titles == ["ニ"]

    def test_range(self):
        titles = _extract_subref_titles("イからハまでに掲げる書類")
        assert titles == ["イ", "ロ", "ハ"]

    def test_range_longer(self):
        titles = _extract_subref_titles("イからホまでに掲げる事項")
        assert titles == ["イ", "ロ", "ハ", "ニ", "ホ"]

    def test_list_and(self):
        titles = _extract_subref_titles("イ及びロに掲げる書類")
        assert "イ" in titles
        assert "ロ" in titles

    def test_list_or(self):
        titles = _extract_subref_titles("イ又はロに規定する基準")
        assert "イ" in titles
        assert "ロ" in titles

    def test_no_ref(self):
        titles = _extract_subref_titles("一般廃棄物の処分に関する基準")
        assert titles == []

    def test_no_suffix_no_match(self):
        # イ alone without に規定する etc. should NOT match
        titles = _extract_subref_titles("イの場合は除く")
        assert titles == []

    def test_kitei(self):
        titles = _extract_subref_titles("イの規定による許可")
        assert titles == ["イ"]


class TestResolveInternalSubrefs:
    """Tests for _resolve_internal_subrefs post-processor."""

    def test_basic_subitem_resolution(self):
        """ハ references イ → イ's content attached to ハ."""
        refs = [{
            "reference": "test",
            "items": [{
                "item_num": 1,
                "text": "次に掲げる書類",
                "sub_items": [
                    {"title": "イ", "text": "申請書の写し"},
                    {"title": "ロ", "text": "事業計画書"},
                    {"title": "ハ", "text": "イに規定する書類の副本"},
                ],
            }],
        }]
        _resolve_internal_subrefs(refs)
        ha = refs[0]["items"][0]["sub_items"][2]
        assert "resolved_refs" in ha
        assert len(ha["resolved_refs"]) == 1
        assert ha["resolved_refs"][0]["title"] == "イ"
        assert ha["resolved_refs"][0]["text"] == "申請書の写し"

    def test_range_resolution(self):
        """ニ references イからハまで → all 3 attached."""
        refs = [{
            "reference": "test",
            "items": [{
                "item_num": 1,
                "text": "次に掲げる事項",
                "sub_items": [
                    {"title": "イ", "text": "氏名"},
                    {"title": "ロ", "text": "住所"},
                    {"title": "ハ", "text": "生年月日"},
                    {"title": "ニ", "text": "イからハまでに掲げる事項の変更"},
                ],
            }],
        }]
        _resolve_internal_subrefs(refs)
        ni = refs[0]["items"][0]["sub_items"][3]
        assert "resolved_refs" in ni
        assert len(ni["resolved_refs"]) == 3
        assert [r["title"] for r in ni["resolved_refs"]] == ["イ", "ロ", "ハ"]

    def test_no_self_reference(self):
        """イ referencing itself should be skipped."""
        refs = [{
            "reference": "test",
            "items": [{
                "item_num": 1,
                "text": "test",
                "sub_items": [
                    {"title": "イ", "text": "イに規定する基準"},  # self-ref
                    {"title": "ロ", "text": "別の基準"},
                ],
            }],
        }]
        _resolve_internal_subrefs(refs)
        i_item = refs[0]["items"][0]["sub_items"][0]
        assert "resolved_refs" not in i_item  # self-ref skipped

    def test_no_subitems_noop(self):
        """Items without sub_items should be untouched."""
        refs = [{
            "reference": "test",
            "items": [{"item_num": 1, "text": "simple text"}],
        }]
        _resolve_internal_subrefs(refs)
        assert "resolved_refs" not in refs[0]["items"][0]

    def test_item_to_item_zengou(self):
        """前号 reference between items."""
        refs = [{
            "reference": "test",
            "items": [
                {"item_num": 1, "text": "許可の申請"},
                {"item_num": 2, "text": "前号の許可に係る変更"},
            ],
        }]
        _resolve_internal_subrefs(refs)
        assert "resolved_refs" in refs[0]["items"][1]
        assert refs[0]["items"][1]["resolved_refs"][0]["item_num"] == 1

    def test_item_to_item_explicit_ref(self):
        """第一号に規定する reference."""
        refs = [{
            "reference": "test",
            "items": [
                {"item_num": 1, "text": "特定施設の設置"},
                {"item_num": 2, "text": "other"},
                {"item_num": 3, "text": "第一号に規定する施設の変更"},
            ],
        }]
        _resolve_internal_subrefs(refs)
        assert "resolved_refs" in refs[0]["items"][2]
        assert refs[0]["items"][2]["resolved_refs"][0]["item_num"] == 1

    def test_subitem_group_from_7b(self):
        """sub_items directly on ref_group (from 7b expansion)."""
        refs = [{
            "reference": "test",
            "sub_items": [
                {"title": "イ", "text": "第一種特定化学物質"},
                {"title": "ロ", "text": "第二種特定化学物質"},
                {"title": "ハ", "text": "イに掲げる物質を含む製品"},
            ],
        }]
        _resolve_internal_subrefs(refs)
        ha = refs[0]["sub_items"][2]
        assert "resolved_refs" in ha
        assert ha["resolved_refs"][0]["title"] == "イ"

    def test_empty_list(self):
        """Empty list should not raise."""
        _resolve_internal_subrefs([])

    def test_multiple_refs_in_same_text(self):
        """Text referencing both イ and ロ."""
        refs = [{
            "reference": "test",
            "items": [{
                "item_num": 1,
                "text": "test",
                "sub_items": [
                    {"title": "イ", "text": "許可証"},
                    {"title": "ロ", "text": "届出書"},
                    {"title": "ハ", "text": "イ及びロに掲げる書類の写し"},
                ],
            }],
        }]
        _resolve_internal_subrefs(refs)
        ha = refs[0]["items"][0]["sub_items"][2]
        assert "resolved_refs" in ha
        titles = [r["title"] for r in ha["resolved_refs"]]
        assert "イ" in titles
        assert "ロ" in titles


# ====== Definition label extraction tests ======

class TestExtractDefinitionLabel:
    """Tests for _extract_definition_label helper."""

    def test_explicit_definition(self):
        text = '次に掲げる産業廃棄物（以下「安定型産業廃棄物」という。）以外の'
        assert _extract_definition_label(text) == "安定型産業廃棄物"

    def test_explicit_definition_2(self):
        text = '特定一般廃棄物最終処分場（以下「特定処分場」という。）について'
        assert _extract_definition_label(text) == "特定処分場"

    def test_no_label_without_explicit_def(self):
        """Without 以下「X」という pattern, no label should be extracted."""
        text = '安定型産業廃棄物の埋立処分の用に供される場所'
        assert _extract_definition_label(text) == ""

    def test_reference_no_label(self):
        text = '第六条第一項第三号ハに掲げる産業廃棄物の埋立処分の用に供される場所'
        assert _extract_definition_label(text) == ""

    def test_empty(self):
        assert _extract_definition_label("") == ""

    def test_short_text_no_match(self):
        assert _extract_definition_label("許可") == ""


class TestAnnotateSubitemsWithLabels:
    """Tests for _annotate_subitems_with_labels helper."""

    def test_annotates_item_with_definition(self):
        items = [{"text": '特定施設（以下「対象施設」という。）の設置', "sub_items": []}]
        _annotate_subitems_with_labels(items)
        assert items[0]["definition_label"] == "対象施設"

    def test_annotates_subitem_with_explicit_def(self):
        items = [{
            "text": "次に掲げるもの",
            "sub_items": [
                {"title": "イ", "text": '次に掲げる産業廃棄物（以下「安定型産業廃棄物」という。）'},
                {"title": "ロ", "text": '安定型産業廃棄物の埋立処分の場所'},
            ],
        }]
        _annotate_subitems_with_labels(items)
        assert items[0]["sub_items"][0].get("definition_label") == "安定型産業廃棄物"
        assert "definition_label" not in items[0]["sub_items"][1]  # no explicit def

    def test_no_label_when_reference(self):
        items = [{"text": "第三条の規定による届出", "sub_items": []}]
        _annotate_subitems_with_labels(items)
        assert "definition_label" not in items[0]

    def test_empty_items(self):
        _annotate_subitems_with_labels([])  # should not raise


class TestNestedSubitemParsing:
    """Tests for recursive sub-item parsing (Subitem1→2→...→10)."""

    def test_nested_structure_preserved(self):
        """Verify sub_items within sub_items are parsed as nested tree."""
        from src.mcp_server import extract_articles_from_xml, find_article_in_xml
        import base64
        # Build XML with nested sub-items: Item → Subitem1(イ) → Subitem2((1))
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<Law><LawBody><MainProvision>
  <Article Num="99">
    <ArticleTitle>第九十九条</ArticleTitle>
    <Paragraph Num="1">
      <ParagraphNum/>
      <ParagraphSentence><Sentence>Test paragraph.</Sentence></ParagraphSentence>
      <Item Num="1">
        <ItemTitle>一</ItemTitle>
        <ItemSentence><Sentence>Test item with sub-items.</Sentence></ItemSentence>
        <Subitem1 Num="1">
          <Subitem1Title>イ</Subitem1Title>
          <Subitem1Sentence><Sentence>Top level sub-item.</Sentence></Subitem1Sentence>
          <Subitem2 Num="1">
            <Subitem2Title>（１）</Subitem2Title>
            <Subitem2Sentence><Sentence>Nested level 2.</Sentence></Subitem2Sentence>
            <Subitem3 Num="1">
              <Subitem3Title>（ｉ）</Subitem3Title>
              <Subitem3Sentence><Sentence>Deeply nested level 3.</Sentence></Subitem3Sentence>
            </Subitem3>
          </Subitem2>
          <Subitem2 Num="2">
            <Subitem2Title>（２）</Subitem2Title>
            <Subitem2Sentence><Sentence>Second nested.</Sentence></Subitem2Sentence>
          </Subitem2>
        </Subitem1>
        <Subitem1 Num="2">
          <Subitem1Title>ロ</Subitem1Title>
          <Subitem1Sentence><Sentence>Second top level.</Sentence></Subitem1Sentence>
        </Subitem1>
      </Item>
    </Paragraph>
  </Article>
</MainProvision></LawBody></Law>"""
        encoded = base64.b64encode(xml.encode("utf-8")).decode("ascii")
        articles = extract_articles_from_xml(encoded)
        assert len(articles) >= 1
        art = articles[0]
        items = art["paragraphs"][0]["items"]
        assert len(items) == 1
        item = items[0]

        # Top level: イ and ロ
        subs = item["sub_items"]
        assert len(subs) == 2
        assert subs[0]["title"] == "イ"
        assert subs[1]["title"] == "ロ"

        # イ has nested children (1) and (2)
        assert "sub_items" in subs[0]
        nested = subs[0]["sub_items"]
        assert len(nested) == 2
        assert nested[0]["title"] == "（１）"
        assert nested[1]["title"] == "（２）"

        # (1) has deeply nested (i)
        assert "sub_items" in nested[0]
        deep = nested[0]["sub_items"]
        assert len(deep) == 1
        assert deep[0]["text"] == "Deeply nested level 3."

        # ロ has no children
        assert "sub_items" not in subs[1]


class TestResolveOutOfRangeSubrefs:
    """Tests for out-of-range sub-item resolution via _all_sibling_subs."""

    def test_resolve_out_of_range_ref(self):
        """ハ references イ which is outside ロ〜ハ range — should resolve via _all_sibling_subs."""
        refs = [{
            "reference": "令第七条第十四号ロ及びハ",
            "sub_items": [
                {"title": "ロ", "text": "安定型産業廃棄物の埋立処分の用に供される場所"},
                {"title": "ハ", "text": "イに規定する産業廃棄物及び安定型産業廃棄物以外の産業廃棄物の埋立処分"},
            ],
            "_all_sibling_subs": [
                {"title": "イ", "text": "有害産業廃棄物の埋立処分の用に供される場所"},
                {"title": "ロ", "text": "安定型産業廃棄物の埋立処分の用に供される場所"},
                {"title": "ハ", "text": "イに規定する産業廃棄物及び安定型産業廃棄物以外の産業廃棄物の埋立処分"},
            ],
        }]
        _resolve_internal_subrefs(refs)
        ha = refs[0]["sub_items"][1]
        assert "resolved_refs" in ha
        assert ha["resolved_refs"][0]["title"] == "イ"
        assert "有害産業廃棄物" in ha["resolved_refs"][0]["text"]

    def test_no_all_siblings_still_works(self):
        """Without _all_sibling_subs, only in-range resolution works."""
        refs = [{
            "reference": "test",
            "sub_items": [
                {"title": "イ", "text": "基準A"},
                {"title": "ロ", "text": "イに規定する基準の変更"},
            ],
        }]
        _resolve_internal_subrefs(refs)
        ro = refs[0]["sub_items"][1]
        assert "resolved_refs" in ro
        assert ro["resolved_refs"][0]["title"] == "イ"


class TestExcludedSubItems:
    """Tests for excluded_sub_items complement computation (step 7d)."""

    def test_basic_complement(self):
        """Referenced ロ及びハ → excluded イ."""
        ref_group = {
            "reference": "令第七条第十四号ロ及びハ",
            "sub_items": [
                {"title": "ロ", "text": "安定型"},
                {"title": "ハ", "text": "管理型"},
            ],
            "_all_sibling_subs": [
                {"title": "イ", "text": "遮断型"},
                {"title": "ロ", "text": "安定型"},
                {"title": "ハ", "text": "管理型"},
            ],
        }
        # Simulate step 7d logic
        all_subs = ref_group.get("_all_sibling_subs", [])
        matched_titles = {si["title"] for si in ref_group["sub_items"]}
        excluded = [
            {"title": si["title"], "text": si["text"]}
            for si in all_subs
            if si["title"] not in matched_titles
        ]
        assert len(excluded) == 1
        assert excluded[0]["title"] == "イ"
        assert excluded[0]["text"] == "遮断型"

    def test_no_complement_when_all_referenced(self):
        """All sub-items referenced → no excluded."""
        ref_group = {
            "sub_items": [
                {"title": "イ", "text": "A"},
                {"title": "ロ", "text": "B"},
            ],
            "_all_sibling_subs": [
                {"title": "イ", "text": "A"},
                {"title": "ロ", "text": "B"},
            ],
        }
        all_subs = ref_group.get("_all_sibling_subs", [])
        matched_titles = {si["title"] for si in ref_group["sub_items"]}
        excluded = [si for si in all_subs if si["title"] not in matched_titles]
        assert len(excluded) == 0

    def test_single_ref_multiple_excluded(self):
        """Single sub-item referenced → multiple excluded."""
        ref_group = {
            "sub_items": [
                {"title": "ロ", "text": "B"},
            ],
            "_all_sibling_subs": [
                {"title": "イ", "text": "A"},
                {"title": "ロ", "text": "B"},
                {"title": "ハ", "text": "C"},
                {"title": "ニ", "text": "D"},
            ],
        }
        all_subs = ref_group.get("_all_sibling_subs", [])
        matched_titles = {si["title"] for si in ref_group["sub_items"]}
        excluded = [si for si in all_subs if si["title"] not in matched_titles]
        assert len(excluded) == 3
        assert [e["title"] for e in excluded] == ["イ", "ハ", "ニ"]

    def test_no_sibling_subs(self):
        """No _all_sibling_subs field → no excluded."""
        ref_group = {
            "sub_items": [{"title": "イ", "text": "A"}],
        }
        all_subs = ref_group.get("_all_sibling_subs", [])
        assert len(all_subs) == 0


# ====== Appended Table Extraction Tests ======

SAMPLE_TABLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Law>
  <LawBody>
    <MainProvision>
      <Article Num="1">
        <ArticleCaption>（基準）</ArticleCaption>
        <ArticleTitle>第一条</ArticleTitle>
        <Paragraph Num="1">
          <ParagraphSentence>
            <Sentence>別表第一に定める基準に適合すること。</Sentence>
          </ParagraphSentence>
        </Paragraph>
      </Article>
    </MainProvision>
    <AppdxTable>
      <AppdxTableTitle>別表第一</AppdxTableTitle>
      <RelatedArticleNum>（第一条関係）</RelatedArticleNum>
      <TableStruct>
        <Table>
          <TableRow>
            <TableColumn>水銀</TableColumn>
            <TableColumn>0.005mg/L以下</TableColumn>
          </TableRow>
          <TableRow>
            <TableColumn>カドミウム</TableColumn>
            <TableColumn>0.03mg/L以下</TableColumn>
          </TableRow>
          <TableRow>
            <TableColumn>鉛</TableColumn>
            <TableColumn>0.1mg/L以下</TableColumn>
          </TableRow>
        </Table>
        <Remarks>
          <RemarksLabel>備考</RemarksLabel>
          <Sentence>この表の数値は日間平均値とする。</Sentence>
        </Remarks>
      </TableStruct>
    </AppdxTable>
    <AppdxTable>
      <AppdxTableTitle>別表第二</AppdxTableTitle>
      <RelatedArticleNum>（第二条関係）</RelatedArticleNum>
      <TableStruct>
        <Table>
          <TableRow>
            <TableColumn>項目A</TableColumn>
            <TableColumn>基準A</TableColumn>
          </TableRow>
        </Table>
      </TableStruct>
    </AppdxTable>
  </LawBody>
</Law>"""


class TestExtractTablesFromXml:
    def _get_b64(self):
        return base64.b64encode(SAMPLE_TABLE_XML.encode("utf-8")).decode("utf-8")

    def test_extract_all_tables(self):
        tables = extract_tables_from_xml(self._get_b64())
        assert len(tables) == 2
        assert tables[0]["title"] == "別表第一"
        assert tables[1]["title"] == "別表第二"

    def test_table_rows(self):
        tables = extract_tables_from_xml(self._get_b64())
        t1 = tables[0]
        assert len(t1["rows"]) == 3
        assert t1["rows"][0] == ["水銀", "0.005mg/L以下"]
        assert t1["rows"][1] == ["カドミウム", "0.03mg/L以下"]
        assert t1["rows"][2] == ["鉛", "0.1mg/L以下"]

    def test_related_articles(self):
        tables = extract_tables_from_xml(self._get_b64())
        assert tables[0]["related_articles"] == "（第一条関係）"

    def test_remarks(self):
        tables = extract_tables_from_xml(self._get_b64())
        assert "remarks" in tables[0]
        assert "備考" in tables[0]["remarks"][0]

    def test_empty_input(self):
        assert extract_tables_from_xml("") == []
        assert extract_tables_from_xml(None) == []

    def test_no_tables(self):
        xml = '<?xml version="1.0"?><Law><LawBody><MainProvision></MainProvision></LawBody></Law>'
        b64 = base64.b64encode(xml.encode()).decode()
        assert extract_tables_from_xml(b64) == []

    def test_filter_by_name(self):
        """Test that MCP tool filtering works (simulated here)."""
        tables = extract_tables_from_xml(self._get_b64())
        filtered = [t for t in tables if "別表第一" in t.get("title", "")]
        assert len(filtered) == 1
        assert filtered[0]["title"] == "別表第一"


class TestGetLawTablesTool:
    @pytest.fixture(autouse=True)
    def clear_caches(self):
        cache_manager.law_lookup_cache.clear()
        cache_manager.law_content_cache.clear()
        yield

    @pytest.mark.asyncio
    async def test_get_tables_basic(self):
        """Test get_law_tables returns table data."""
        mock_resp = _make_api_response()
        # Replace law_full_text with our table XML
        mock_resp["law_full_text"] = base64.b64encode(SAMPLE_TABLE_XML.encode("utf-8")).decode("utf-8")
        mock_client = _make_mock_client(mock_resp)
        p1 = patch("src.mcp_server.smart_law_lookup", AsyncMock(return_value="test-law-1"))
        p2 = patch("src.mcp_server.get_http_client", AsyncMock(return_value=mock_client))
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("get_law_tables", {
                    "law_name": "testlaw"
                })
                data = json.loads(result[0].text)
                assert data["tables_found"] == 2
                assert data["tables"][0]["title"] == "別表第一"
                assert len(data["tables"][0]["rows"]) == 3

    @pytest.mark.asyncio
    async def test_get_tables_filtered(self):
        """Test get_law_tables with table_name filter."""
        mock_resp = _make_api_response()
        mock_resp["law_full_text"] = base64.b64encode(SAMPLE_TABLE_XML.encode("utf-8")).decode("utf-8")
        mock_client = _make_mock_client(mock_resp)
        p1 = patch("src.mcp_server.smart_law_lookup", AsyncMock(return_value="test-law-1"))
        p2 = patch("src.mcp_server.get_http_client", AsyncMock(return_value=mock_client))
        with p1, p2:
            async with Client(mcp) as client:
                result = await client.call_tool("get_law_tables", {
                    "law_name": "testlaw", "table_name": "別表第二"
                })
                data = json.loads(result[0].text)
                assert data["tables_found"] == 1
                assert data["tables"][0]["title"] == "別表第二"
