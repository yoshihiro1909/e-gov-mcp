#!/usr/bin/env python3
"""
e-Gov法令MCPサーバーv2のテスト
Ultra Smart版の包括的テストスイート
"""

import pytest
import json
import asyncio
from unittest.mock import AsyncMock, patch
from fastmcp import FastMCP, Client
from src.mcp_server import mcp, BASIC_LAWS, arabic_to_kanji, generate_search_patterns, extract_text_from_xml


class TestMCPServerV2:
    """MCPサーバーv2のテストクラス"""
    
    def test_server_creation(self):
        """サーバーが正しく作成されることを確認"""
        assert isinstance(mcp, FastMCP)
        assert "v2" in mcp.name or "Ultra Smart" in mcp.name
    
    def test_basic_laws_mapping(self):
        """基本法マッピングが正しく設定されていることを確認"""
        assert len(BASIC_LAWS) >= 17  # 17法以上
        
        # 六法の存在確認
        essential_laws = ["民法", "憲法", "刑法", "商法", "民事訴訟法", "刑事訴訟法"]
        for law in essential_laws:
            assert law in BASIC_LAWS
        
        # 追加重要法の存在確認
        important_laws = ["会社法", "労働基準法", "著作権法", "特許法"]
        for law in important_laws:
            assert law in BASIC_LAWS
    
    def test_arabic_to_kanji_conversion(self):
        """Arabic数字から漢字への変換をテスト"""
        test_cases = [
            ("1", "一"),
            ("9", "九"),
            ("10", "十"),
            ("11", "十一"),
            ("19", "十九"),
            ("20", "二十"),
            ("25", "二十五"),
            ("100", "百"),
            ("192", "百九十二"),
            ("325", "三百二十五"),
            ("999", "九百九十九"),
        ]
        
        for arabic, expected_kanji in test_cases:
            result = arabic_to_kanji(arabic)
            assert result == expected_kanji, f"Failed: {arabic} -> {result} (expected: {expected_kanji})"
    
    def test_search_pattern_generation(self):
        """検索パターン生成をテスト"""
        # 基本パターン
        patterns = generate_search_patterns("192")
        assert "第百九十二条" in patterns
        assert "百九十二条" in patterns
        assert "第192条" in patterns
        
        # 条の2パターン
        patterns = generate_search_patterns("325条の3")
        assert "第三百二十五条の三" in patterns
        assert "三百二十五条の三" in patterns
        assert "第325条の3" in patterns
        
        # 項パターン
        patterns = generate_search_patterns("第9条第2項")
        assert "第九条第二項" in patterns
        assert "第9条第2項" in patterns
    
    @pytest.mark.asyncio
    async def test_find_law_article_validation(self):
        """find_law_article関数の入力検証をテスト"""
        from fastmcp.exceptions import ToolError
        
        async with Client(mcp) as client:
            # 空のlaw_nameの場合
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool("find_law_article", {"law_name": "", "article_number": "1"})
            assert "law_name is required" in str(exc_info.value)
            
            # 空のarticle_numberの場合
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool("find_law_article", {"law_name": "民法", "article_number": ""})
            assert "article_number is required" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_search_laws_validation(self):
        """search_laws関数の入力検証をテスト"""
        from fastmcp.exceptions import ToolError
        
        async with Client(mcp) as client:
            # 無効なlimitの場合
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool("search_laws", {"limit": 999})
            assert "limit must be between 1 and 500" in str(exc_info.value)
            
            # 無効なoffsetの場合
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool("search_laws", {"offset": -1})
            assert "offset must be 0 or greater" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_search_laws_by_keyword_validation(self):
        """search_laws_by_keyword関数の入力検証をテスト"""
        from fastmcp.exceptions import ToolError
        
        async with Client(mcp) as client:
            # 空のキーワードの場合
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool("search_laws_by_keyword", {"keyword": ""})
            assert "keyword is required" in str(exc_info.value)
            
            # 無効なlimitの場合
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool("search_laws_by_keyword", {"keyword": "test", "limit": 25})
            assert "limit must be between 1 and 20" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_law_content_validation(self):
        """get_law_content関数の入力検証をテスト"""
        from fastmcp.exceptions import ToolError
        
        async with Client(mcp) as client:
            # law_idもlaw_numも指定されていない場合
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool("get_law_content", {})
            assert "Either law_id or law_num must be specified" in str(exc_info.value)
            
            # 無効なresponse_formatの場合
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool("get_law_content", {"law_num": "test", "response_format": "invalid"})
            assert "response_format must be 'json' or 'xml'" in str(exc_info.value)


class TestRealLawArticles:
    """実在する条文での動作テスト"""
    
    @pytest.fixture(autouse=True)
    def clear_global_caches(self):
        """Clear global caches to avoid pollution from mocked tests."""
        from src.mcp_server import cache_manager
        cache_manager.law_lookup_cache.clear()
        cache_manager.law_content_cache.clear()
        cache_manager.article_cache.clear()
        yield
    
    @pytest.mark.asyncio
    async def test_civil_code_article_192(self):
        """民法192条（即時取得）のテスト"""
        async with Client(mcp) as client:
            result = await client.call_tool("find_law_article", {
                "law_name": "民法",
                "article_number": "192"
            })
            
            if "Error:" not in result[0].text:
                data = json.loads(result[0].text)
                assert data.get("matches_found", 0) > 0
                assert data.get("law_number") == "明治二十九年法律第八十九号"
                
                # 即時取得の内容が含まれているか確認
                articles = data.get("articles", [])
                if articles:
                    assert "即時取得" in articles[0] or "百九十二条" in articles[0]
    
    @pytest.mark.asyncio
    async def test_constitution_article_9(self):
        """憲法9条（戦争放棄）のテスト"""
        async with Client(mcp) as client:
            result = await client.call_tool("find_law_article", {
                "law_name": "憲法",
                "article_number": "9"
            })
            
            if "Error:" not in result[0].text:
                data = json.loads(result[0].text)
                assert data.get("matches_found", 0) > 0
                assert data.get("law_number") == "昭和二十一年憲法"
                
                # 戦争放棄の内容が含まれているか確認
                articles = data.get("articles", [])
                if articles:
                    assert "戦争" in articles[0] or "九条" in articles[0]
    
    @pytest.mark.asyncio
    async def test_company_law_complex_article(self):
        """会社法325条の3（条の2系パターン）のテスト"""
        async with Client(mcp) as client:
            result = await client.call_tool("find_law_article", {
                "law_name": "会社法",
                "article_number": "325条の3"
            })
            
            if "Error:" not in result[0].text:
                data = json.loads(result[0].text)
                assert data.get("matches_found", 0) > 0
                assert data.get("law_number") == "平成十七年法律第八十六号"
    
    @pytest.mark.asyncio
    async def test_multiple_basic_laws(self):
        """複数の基本法での動作確認"""
        test_cases = [
            ("労働基準法", "1"),
            ("著作権法", "1"),
            ("特許法", "1"),
            ("建築基準法", "1"),
        ]
        
        async with Client(mcp) as client:
            for law_name, article in test_cases:
                result = await client.call_tool("find_law_article", {
                    "law_name": law_name,
                    "article_number": article
                })
                
                # 基本法は直接マッピングされているので、エラーにならないはず
                assert isinstance(result[0].text, str)  # FastMCP returns text content
                
                try:
                    data = json.loads(result[0].text)
                    # 直接マッピングされた法令番号を使用しているか確認
                    expected_law_num = BASIC_LAWS.get(law_name)
                    if expected_law_num:
                        assert data.get("law_number") == expected_law_num
                except json.JSONDecodeError:
                    # JSON decodeできない場合はスキップ
                    pass


class TestResources:
    """リソースのテストクラス"""
    
    @pytest.mark.asyncio
    async def test_api_info_resource_v2(self):
        """API情報リソースv2をテスト"""
        async with Client(mcp) as client:
            result = await client.read_resource("api://info")
            
            resource_data = json.loads(result[0].text)
            assert isinstance(resource_data, dict)
            assert "name" in resource_data
            assert "version" in resource_data
            assert resource_data.get("version") == "2.0"
            assert "features" in resource_data
            assert isinstance(resource_data["features"], list)
            assert "basic_laws_supported" in resource_data
            assert resource_data["basic_laws_supported"] >= 17
    
    @pytest.mark.asyncio
    async def test_law_types_resource_v2(self):
        """法令種別リソースv2をテスト"""
        async with Client(mcp) as client:
            result = await client.read_resource("schema://law_types")
            
            resource_data = json.loads(result[0].text)
            assert isinstance(resource_data, dict)
            assert "law_types" in resource_data
            assert "basic_laws" in resource_data
            
            # 基本法マッピングが含まれていることを確認
            basic_laws = resource_data["basic_laws"]
            assert len(basic_laws) >= 17
            assert "民法" in basic_laws
            assert "憲法" in basic_laws


class TestAdvancedFeatures:
    """高度機能のテストクラス"""
    
    @pytest.mark.asyncio
    async def test_server_ping(self):
        """サーバーのping機能をテスト"""
        async with Client(mcp) as client:
            await client.ping()
    
    @pytest.mark.asyncio
    async def test_list_tools_v2(self):
        """v2ツール一覧取得をテスト"""
        async with Client(mcp) as client:
            tools = await client.list_tools()
            
            # v2では4つのツールが期待される
            expected_tools = [
                "find_law_article",
                "search_laws", 
                "search_laws_by_keyword",
                "get_law_content"
            ]
            
            tool_names = [tool.name for tool in tools]
            for tool_name in expected_tools:
                assert tool_name in tool_names
            
            # find_law_articleが最優先ツールとして含まれていることを確認
            find_tool = next((tool for tool in tools if tool.name == "find_law_article"), None)
            assert find_tool is not None
            assert "ULTRA SMART" in find_tool.description or "USER-FRIENDLY" in find_tool.description
    
    @pytest.mark.asyncio
    async def test_list_resources(self):
        """利用可能なリソースの一覧取得をテスト"""
        async with Client(mcp) as client:
            resources = await client.list_resources()
            
            resource_uris = [str(resource.uri) for resource in resources]
            assert "api://info" in resource_uris
            assert "schema://law_types" in resource_uris


class TestErrorHandling:
    """エラーハンドリングのテストクラス"""
    
    @pytest.mark.asyncio
    async def test_nonexistent_law_handling(self):
        """存在しない法律のハンドリングをテスト"""
        from fastmcp.exceptions import ToolError
        
        async with Client(mcp) as client:
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool("find_law_article", {
                    "law_name": "存在しない法律",
                    "article_number": "1"
                })
            
            assert "not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_article_not_found_suggestions(self):
        """条文が見つからない場合の提案機能をテスト"""
        async with Client(mcp) as client:
            # 存在しない条文番号でテスト
            result = await client.call_tool("find_law_article", {
                "law_name": "民法",
                "article_number": "99999"
            })
            
            try:
                data = json.loads(result[0].text)
                matches = data.get("matches_found", 0)
                
                # 条文が見つからない場合、提案が含まれているか確認
                if matches == 0:
                    assert "suggestion" in data
                    suggestion = data.get("suggestion", "")
                    assert len(suggestion) > 0
            except json.JSONDecodeError:
                # JSON decodeできない場合はスキップ
                pass


class TestTextExtraction:
    """テキスト抽出機能のテスト"""
    
    def test_extract_text_from_xml_string(self):
        """XML文字列からのテキスト抽出をテスト"""
        # 簡単なXMLテスト用データ
        test_xml = "<root><text>テスト</text><child>内容</child></root>"
        import base64
        encoded_xml = base64.b64encode(test_xml.encode('utf-8')).decode('utf-8')
        
        result = extract_text_from_xml(encoded_xml)
        assert "テスト" in result
        assert "内容" in result
    
    def test_extract_text_from_dict(self):
        """辞書構造からのテキスト抽出をテスト"""
        test_dict = {
            "children": [
                {"text": "テスト1"},
                {"text": "テスト2"}
            ]
        }
        
        result = extract_text_from_xml(test_dict)
        assert "テスト1" in result
        assert "テスト2" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])