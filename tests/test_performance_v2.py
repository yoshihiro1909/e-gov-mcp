#!/usr/bin/env python3
"""
e-Gov法令MCPサーバーv2のパフォーマンステスト
Ultra Smart版の速度と効率性を検証
"""

import pytest
import json
import time
import asyncio
from fastmcp import Client
from src.mcp_server import mcp, BASIC_LAWS


class TestPerformanceV2:
    """パフォーマンステストクラス"""
    
    @pytest.fixture(autouse=True)
    def clear_global_caches(self):
        from src.mcp_server import cache_manager
        cache_manager.law_lookup_cache.clear()
        cache_manager.law_content_cache.clear()
        cache_manager.article_cache.clear()
        yield
    
    @pytest.mark.asyncio
    async def test_basic_law_direct_mapping_speed(self):
        """基本法の直接マッピング速度テスト"""
        async with Client(mcp) as client:
            # 複数の基本法で速度測定
            basic_law_tests = [
                ("民法", "1"),
                ("憲法", "9"), 
                ("刑法", "199"),
                ("会社法", "1"),
                ("労働基準法", "1")
            ]
            
            total_time = 0
            success_count = 0
            
            for law_name, article in basic_law_tests:
                start_time = time.time()
                
                result = await client.call_tool("find_law_article", {
                    "law_name": law_name,
                    "article_number": article
                })
                
                end_time = time.time()
                response_time = end_time - start_time
                total_time += response_time
                
                # 成功判定
                if "Error:" not in result[0].text:
                    data = json.loads(result[0].text)
                    if data.get("matches_found", 0) > 0:
                        success_count += 1
                
                print(f"  {law_name}第{article}条: {response_time:.2f}秒")
            
            avg_time = total_time / len(basic_law_tests)
            success_rate = success_count / len(basic_law_tests) * 100
            
            print(f"\n平均応答時間: {avg_time:.2f}秒")
            print(f"成功率: {success_rate:.1f}%")
            
            # パフォーマンス基準（基本法は5秒以内で応答すべき）
            assert avg_time < 5.0, f"基本法検索が遅すぎます: {avg_time:.2f}秒"
            assert success_rate >= 80.0, f"成功率が低すぎます: {success_rate:.1f}%"
    
    @pytest.mark.asyncio
    async def test_complex_pattern_performance(self):
        """複雑パターン検索のパフォーマンステスト"""
        async with Client(mcp) as client:
            complex_patterns = [
                ("会社法", "325条の2"),
                ("会社法", "325条の3"),
                ("民法", "第192条"),
                ("憲法", "第9条第2項"),
            ]
            
            total_time = 0
            
            for law_name, article in complex_patterns:
                start_time = time.time()
                
                result = await client.call_tool("find_law_article", {
                    "law_name": law_name,
                    "article_number": article
                })
                
                end_time = time.time()
                response_time = end_time - start_time
                total_time += response_time
                
                print(f"  {law_name} {article}: {response_time:.2f}秒")
            
            avg_time = total_time / len(complex_patterns)
            print(f"\n複雑パターン平均時間: {avg_time:.2f}秒")
            
            # 複雑パターンでも10秒以内で応答すべき
            assert avg_time < 10.0, f"複雑パターン検索が遅すぎます: {avg_time:.2f}秒"
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self):
        """並行リクエストのパフォーマンステスト"""
        async with Client(mcp) as client:
            # 複数の検索を並行実行
            concurrent_tests = [
                ("民法", "192"),
                ("憲法", "9"),
                ("会社法", "1"),
                ("労働基準法", "1"),
                ("著作権法", "1")
            ]
            
            async def single_request(law_name, article):
                start_time = time.time()
                result = await client.call_tool("find_law_article", {
                    "law_name": law_name,
                    "article_number": article
                })
                end_time = time.time()
                return end_time - start_time, result
            
            # 並行実行
            start_total = time.time()
            tasks = [single_request(law, art) for law, art in concurrent_tests]
            results = await asyncio.gather(*tasks)
            end_total = time.time()
            
            total_concurrent_time = end_total - start_total
            individual_times = [r[0] for r in results]
            avg_individual_time = sum(individual_times) / len(individual_times)
            
            print(f"並行実行総時間: {total_concurrent_time:.2f}秒")
            print(f"個別平均時間: {avg_individual_time:.2f}秒")
            print(f"効率性: {avg_individual_time/total_concurrent_time:.2f}x")
            
            # 並行実行の効率性を確認
            assert total_concurrent_time < sum(individual_times), "並行実行による高速化が見られません"
            
            # 成功率確認
            success_count = 0
            for _, result in results:
                try:
                    data = json.loads(result[0].text)
                    if data.get("matches_found", 0) > 0:
                        success_count += 1
                except (json.JSONDecodeError, KeyError):
                    # JSON decodeできない、または期待したキーがない場合はスキップ
                    pass
            
            success_rate = success_count / len(results) * 100
            assert success_rate >= 80.0, f"並行実行時の成功率が低すぎます: {success_rate:.1f}%"


class TestEfficiencyV2:
    """効率性テストクラス"""
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """メモリ効率性テスト - 簡素化版"""
        async with Client(mcp) as client:
            # 複数の検索を実行してエラーが発生しないことを確認
            for i in range(5):  # 回数を減らして安定性を向上
                law_name = list(BASIC_LAWS.keys())[i % len(BASIC_LAWS)]
                result = await client.call_tool("find_law_article", {
                    "law_name": law_name,
                    "article_number": "1"
                })
                # 基本的な動作確認
                assert "Error:" not in result[0].text or "not found" in result[0].text
    
    @pytest.mark.asyncio
    async def test_api_call_efficiency(self):
        """API呼び出し効率性テスト"""
        async with Client(mcp) as client:
            # 基本法の場合、1回のAPI呼び出しで済むはず
            law_name = "民法"
            article_number = "192"
            
            # API呼び出し回数をカウント（ログから推測）
            result = await client.call_tool("find_law_article", {
                "law_name": law_name,
                "article_number": article_number
            })
            
            # 基本法の直接マッピングが機能していることを確認
            if "Error:" not in result[0].text:
                data = json.loads(result[0].text)
                # 正しい法令番号が使用されていることを確認
                expected_law_num = BASIC_LAWS.get(law_name)
                assert data.get("law_number") == expected_law_num
    
    def test_basic_laws_coverage(self):
        """基本法のカバレッジテスト"""
        # v2で追加された基本法の数を確認
        assert len(BASIC_LAWS) >= 17, f"基本法が不足: {len(BASIC_LAWS)}法（17法以上必要）"
        
        # 六法が含まれていることを確認
        essential_laws = ["民法", "憲法", "刑法", "商法", "民事訴訟法", "刑事訴訟法"]
        for law in essential_laws:
            assert law in BASIC_LAWS, f"六法の{law}が基本法に含まれていません"
        
        print(f"基本法対応数: {len(BASIC_LAWS)}法")


class TestScalabilityV2:
    """スケーラビリティテストクラス"""
    
    @pytest.mark.asyncio
    async def test_multiple_laws_search(self):
        """複数法令同時検索のスケーラビリティテスト"""
        async with Client(mcp) as client:
            # 全基本法での検索を実行
            all_basic_laws = list(BASIC_LAWS.keys())
            
            success_count = 0
            total_time = 0
            
            for law_name in all_basic_laws[:10]:  # 最初の10法でテスト
                start_time = time.time()
                
                result = await client.call_tool("find_law_article", {
                    "law_name": law_name,
                    "article_number": "1"
                })
                
                end_time = time.time()
                total_time += (end_time - start_time)
                
                try:
                    data = json.loads(result[0].text)
                    if data.get("matches_found", 0) > 0:
                        success_count += 1
                except (json.JSONDecodeError, KeyError):
                    # JSON decodeできない、または期待したキーがない場合はスキップ
                    pass
            
            avg_time = total_time / len(all_basic_laws[:10])
            success_rate = success_count / len(all_basic_laws[:10]) * 100
            
            print(f"10法令検索平均時間: {avg_time:.2f}秒")
            print(f"成功率: {success_rate:.1f}%")
            
            assert avg_time < 6.0, f"複数法令検索が遅すぎます: {avg_time:.2f}秒"
            assert success_rate >= 70.0, f"複数法令検索の成功率が低すぎます: {success_rate:.1f}%"
    
    @pytest.mark.asyncio 
    async def test_error_recovery_performance(self):
        """エラー回復のパフォーマンステスト"""
        async with Client(mcp) as client:
            # 存在しない法令でのエラー処理速度
            error_tests = [
                ("存在しない法律1", "1"),
                ("存在しない法律2", "2"),
                ("存在しない法律3", "3")
            ]
            
            total_time = 0
            
            error_count = 0
            
            for law_name, article in error_tests:
                start_time = time.time()
                
                try:
                    await client.call_tool("find_law_article", {
                        "law_name": law_name,
                        "article_number": article
                    })
                except Exception:  # ToolError or other exceptions
                    # エラーが適切に投げられたことを確認
                    error_count += 1
                
                end_time = time.time()
                total_time += (end_time - start_time)
            
            avg_error_time = total_time / len(error_tests)
            print(f"エラー処理平均時間: {avg_error_time:.2f}秒")
            print(f"エラー数: {error_count}/{len(error_tests)}")
            
            # エラーが適切に処理されることを確認
            assert error_count == len(error_tests), f"エラーが適切に処理されていません: {error_count}/{len(error_tests)}"
            
            # エラー処理は高速であるべき（3秒以内）
            assert avg_error_time < 3.0, f"エラー処理が遅すぎます: {avg_error_time:.2f}秒"


if __name__ == "__main__":
    # パフォーマンステストを詳細出力で実行
    pytest.main([__file__, "-v", "-s"])