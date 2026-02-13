#!/usr/bin/env python3
"""Tests for new features: XML extraction, normalization, related laws, caching."""

import pytest
import base64
import xml.etree.ElementTree as ET
from src.mcp_server import (
    extract_articles_from_xml,
    find_article_in_xml,
    _normalize_number,
    _normalize_article_number,
    arabic_to_kanji,
    config_loader,
)


# Minimal law XML for testing
SAMPLE_LAW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Law>
  <LawBody>
    <MainProvision>
      <Article Num="1">
        <ArticleCaption>（目的）</ArticleCaption>
        <ArticleTitle>第一条</ArticleTitle>
        <Paragraph Num="1">
          <ParagraphNum/>
          <ParagraphSentence><Sentence>この法律は、テスト目的の法律である。</Sentence></ParagraphSentence>
        </Paragraph>
        <Paragraph Num="2">
          <ParagraphNum>２</ParagraphNum>
          <ParagraphSentence><Sentence>第二項のテスト文。</Sentence></ParagraphSentence>
          <Item Num="1">
            <ItemTitle>一</ItemTitle>
            <ItemSentence><Sentence>第一号のテスト。</Sentence></ItemSentence>
          </Item>
          <Item Num="2">
            <ItemTitle>二</ItemTitle>
            <ItemSentence><Sentence>第二号のテスト。</Sentence></ItemSentence>
          </Item>
        </Paragraph>
      </Article>
      <Article Num="2">
        <ArticleCaption>（定義）</ArticleCaption>
        <ArticleTitle>第二条</ArticleTitle>
        <Paragraph Num="1">
          <ParagraphNum/>
          <ParagraphSentence><Sentence>この法律において定義テスト。</Sentence></ParagraphSentence>
        </Paragraph>
      </Article>
      <Article Num="6_2">
        <ArticleCaption>（特例）</ArticleCaption>
        <ArticleTitle>第六条の二</ArticleTitle>
        <Paragraph Num="1">
          <ParagraphNum/>
          <ParagraphSentence><Sentence>第六条の二のテスト。</Sentence></ParagraphSentence>
        </Paragraph>
      </Article>
    </MainProvision>
    <SupplProvision>
      <Article Num="1">
        <ArticleTitle>附則第一条</ArticleTitle>
        <Paragraph Num="1">
          <ParagraphNum/>
          <ParagraphSentence><Sentence>この法律は公布の日から施行する。</Sentence></ParagraphSentence>
        </Paragraph>
      </Article>
    </SupplProvision>
  </LawBody>
</Law>"""

SAMPLE_B64 = base64.b64encode(SAMPLE_LAW_XML.encode('utf-8')).decode('ascii')


class TestExtractArticlesFromXml:
    """extract_articles_from_xml のテスト"""

    def test_basic_extraction(self):
        articles = extract_articles_from_xml(SAMPLE_B64)
        # 本則3 + 附則1 = 4
        assert len(articles) == 4

    def test_main_provision_tagged(self):
        articles = extract_articles_from_xml(SAMPLE_B64)
        main_articles = [a for a in articles if a['source'] == 'main']
        suppl_articles = [a for a in articles if a['source'] == 'suppl']
        assert len(main_articles) == 3
        assert len(suppl_articles) == 1

    def test_article_num_attribute(self):
        articles = extract_articles_from_xml(SAMPLE_B64)
        nums = [a['num'] for a in articles if a['source'] == 'main']
        assert '1' in nums
        assert '2' in nums
        assert '6_2' in nums

    def test_paragraphs_extracted(self):
        articles = extract_articles_from_xml(SAMPLE_B64)
        art1 = [a for a in articles if a['num'] == '1' and a['source'] == 'main'][0]
        assert len(art1['paragraphs']) == 2
        assert art1['paragraphs'][0]['num'] == '1'
        assert art1['paragraphs'][1]['num'] == '2'

    def test_items_extracted(self):
        articles = extract_articles_from_xml(SAMPLE_B64)
        art1 = [a for a in articles if a['num'] == '1' and a['source'] == 'main'][0]
        para2 = art1['paragraphs'][1]
        assert len(para2['items']) == 2

    def test_caption_extracted(self):
        articles = extract_articles_from_xml(SAMPLE_B64)
        art1 = [a for a in articles if a['num'] == '1' and a['source'] == 'main'][0]
        assert '目的' in art1['caption']

    def test_invalid_input(self):
        assert extract_articles_from_xml("not-base64!!!") == []
        assert extract_articles_from_xml(None) == []
        assert extract_articles_from_xml({"dict": True}) == []


class TestFindArticleInXml:
    """find_article_in_xml のテスト"""

    def test_find_by_number(self):
        results = find_article_in_xml(SAMPLE_B64, "1")
        assert len(results) == 1
        assert '目的' in results[0]['full_text']

    def test_find_compound_number(self):
        results = find_article_in_xml(SAMPLE_B64, "6の2")
        assert len(results) == 1
        assert '特例' in results[0]['full_text']

    def test_find_with_prefix(self):
        results = find_article_in_xml(SAMPLE_B64, "第2条")
        assert len(results) == 1
        assert '定義' in results[0]['full_text']

    def test_main_only_default(self):
        # Article 1 exists in both main and suppl; main_only=True should return only main
        results = find_article_in_xml(SAMPLE_B64, "1", main_only=True)
        assert len(results) == 1
        assert 'テスト目的' in results[0]['full_text']

    def test_include_suppl(self):
        results = find_article_in_xml(SAMPLE_B64, "1", main_only=False)
        assert len(results) == 2  # main + suppl

    def test_paragraph_filter(self):
        results = find_article_in_xml(SAMPLE_B64, "1", paragraph="2")
        assert len(results) == 1
        assert results[0].get('filtered') is True
        assert len(results[0]['paragraphs']) == 1
        assert results[0]['paragraphs'][0]['num'] == '2'

    def test_item_filter(self):
        results = find_article_in_xml(SAMPLE_B64, "1", paragraph="2", item="1")
        assert len(results) == 1
        para = results[0]['paragraphs'][0]
        assert len(para['items']) == 1
        assert para['items'][0]['num'] == '1'

    def test_not_found(self):
        results = find_article_in_xml(SAMPLE_B64, "999")
        assert len(results) == 0

    def test_fullwidth_input(self):
        results = find_article_in_xml(SAMPLE_B64, "２")
        assert len(results) == 1

    def test_compound_pattern(self):
        """第1条第2項第1号 pattern"""
        results = find_article_in_xml(SAMPLE_B64, "第1条第2項第1号")
        assert len(results) == 1
        assert results[0].get('filtered_paragraph') == '2'
        assert results[0].get('filtered_item') == '1'


class TestNormalizeNumber:
    """_normalize_number のテスト"""

    def test_fullwidth_to_halfwidth(self):
        assert _normalize_number("１２３") == "123"

    def test_mixed(self):
        assert _normalize_number("第１２条") == "第12条"

    def test_empty(self):
        assert _normalize_number("") == ""

    def test_none(self):
        assert _normalize_number(None) is None

    def test_already_halfwidth(self):
        assert _normalize_number("456") == "456"


class TestNormalizeArticleNumber:
    """_normalize_article_number のテスト"""

    def test_basic(self):
        assert _normalize_article_number("6") == "6"

    def test_no_separator(self):
        assert _normalize_article_number("6の2") == "6_2"

    def test_dash_separator(self):
        assert _normalize_article_number("6-2") == "6_2"

    def test_long_dash_separator(self):
        assert _normalize_article_number("6ー2") == "6_2"

    def test_em_dash_separator(self):
        assert _normalize_article_number("6─2") == "6_2"

    def test_fullwidth_digits(self):
        assert _normalize_article_number("６の２") == "6_2"

    def test_strip_prefix_suffix(self):
        assert _normalize_article_number("第6条") == "6"

    def test_compound_with_prefix(self):
        assert _normalize_article_number("第6条の2") == "6_2"

    def test_double_compound(self):
        """第9条の2の2 → 9_2_2"""
        assert _normalize_article_number("9の2の2") == "9_2_2"

    def test_double_compound_with_prefix(self):
        """第14条の3の2 → 14_3_2"""
        assert _normalize_article_number("第14条の3の2") == "14_3_2"

    def test_triple_compound(self):
        """9の3の3 → 9_3_3"""
        assert _normalize_article_number("9の3の3") == "9_3_3"


class TestFindArticleCompoundNumber:
    """Test that find_article_in_xml matches double/triple compound article numbers."""

    COMPOUND_XML = base64.b64encode("""<?xml version="1.0" encoding="UTF-8"?>
<Law><LawBody><MainProvision>
  <Article Num="9_2"><ArticleTitle>第九条の二</ArticleTitle>
    <Paragraph Num="1"><ParagraphNum/><ParagraphSentence><Sentence>改善命令。</Sentence></ParagraphSentence></Paragraph>
  </Article>
  <Article Num="9_2_2"><ArticleTitle>第九条の二の二</ArticleTitle>
    <Paragraph Num="1"><ParagraphNum/><ParagraphSentence><Sentence>許可の取消し。</Sentence></ParagraphSentence></Paragraph>
  </Article>
  <Article Num="14_3"><ArticleTitle>第十四条の三</ArticleTitle>
    <Paragraph Num="1"><ParagraphNum/><ParagraphSentence><Sentence>事業の停止。</Sentence></ParagraphSentence></Paragraph>
  </Article>
  <Article Num="14_3_2"><ArticleTitle>第十四条の三の二</ArticleTitle>
    <Paragraph Num="1"><ParagraphNum/><ParagraphSentence><Sentence>許可の取消し。</Sentence></ParagraphSentence></Paragraph>
  </Article>
</MainProvision></LawBody></Law>""".encode()).decode()

    def test_match_9_2_2(self):
        matches = find_article_in_xml(self.COMPOUND_XML, "9の2の2")
        assert len(matches) == 1
        assert matches[0]["num"] == "9_2_2"

    def test_match_14_3_2(self):
        matches = find_article_in_xml(self.COMPOUND_XML, "14の3の2")
        assert len(matches) == 1
        assert matches[0]["num"] == "14_3_2"

    def test_match_9_2_not_9_2_2(self):
        """9の2 should match 9_2, NOT 9_2_2."""
        matches = find_article_in_xml(self.COMPOUND_XML, "9の2")
        assert len(matches) == 1
        assert matches[0]["num"] == "9_2"

    def test_match_14_3_not_14_3_2(self):
        """14の3 should match 14_3, NOT 14_3_2."""
        matches = find_article_in_xml(self.COMPOUND_XML, "14の3")
        assert len(matches) == 1
        assert matches[0]["num"] == "14_3"


class TestConfigLoaderRelatedLaws:
    """ConfigLoader.related_laws のテスト"""

    def test_related_laws_loaded(self):
        related = config_loader.related_laws
        assert isinstance(related, dict)

    def test_reload(self):
        config_loader.reload_config()
        # Should reload without error
        assert config_loader.basic_laws is not None


class TestDirectMappings:
    """Test that direct mappings resolve correctly for 施行令/施行規則."""

    @pytest.mark.parametrize("alias,expected_formal", [
        ("建基法施行令", "建築基準法施行令"),
        ("道交法施行令", "道路交通法施行令"),
        ("個情法施行令", "個人情報の保護に関する法律施行令"),
        ("労基法施行規則", "労働基準法施行規則"),
        ("建基法施行規則", "建築基準法施行規則"),
        ("道交法施行規則", "道路交通法施行規則"),
        ("個情法施行規則", "個人情報の保護に関する法律施行規則"),
    ])
    def test_alias_to_formal(self, alias, expected_formal):
        aliases = config_loader.law_aliases
        assert aliases.get(alias) == expected_formal

    @pytest.mark.parametrize("formal_name", [
        "建築基準法施行令",
        "道路交通法施行令",
        "会社法施行令",
        "所得税法施行令",
        "著作権法施行令",
        "個人情報の保護に関する法律施行令",
        "建築基準法施行規則",
        "道路交通法施行規則",
        "会社法施行規則",
        "所得税法施行規則",
        "著作権法施行規則",
        "個人情報の保護に関する法律施行規則",
        "労働基準法施行規則",
    ])
    def test_direct_mapping_exists(self, formal_name):
        basic = config_loader.basic_laws
        assert formal_name in basic, f"{formal_name} not in basic_laws"
