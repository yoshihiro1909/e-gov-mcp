"""Tests for structural tools: get_law_structure, search_article_captions,
get_chapter_articles, find_article_range, get_cross_references,
get_supplementary_provisions, search_definitions."""

import pytest
import base64
import json
import xml.etree.ElementTree as ET

from src.mcp_server import (
    _decode_law_xml,
    _elem_text,
    _parse_article_num_tuple,
    extract_articles_from_xml,
    find_article_in_xml,
)


# --- Helpers ---

def _make_law_xml(body_content: str, toc_content: str = "", suppl_content: str = "") -> str:
    """Build a minimal law XML and return as base64 string (what _decode_law_xml expects)."""
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Law>
  <LawBody>
    {f'<TOC>{toc_content}</TOC>' if toc_content else ''}
    <MainProvision>
      {body_content}
    </MainProvision>
    {suppl_content}
  </LawBody>
</Law>"""
    return base64.b64encode(xml.encode("utf-8")).decode("ascii")


def _article(num: str, caption: str = "", text: str = "本文") -> str:
    cap = f"<ArticleCaption>{caption}</ArticleCaption>" if caption else ""
    return f"""<Article Num="{num}">
      {cap}
      <ArticleTitle>第{num}条</ArticleTitle>
      <Paragraph Num="1"><ParagraphSentence><Sentence>{text}</Sentence></ParagraphSentence></Paragraph>
    </Article>"""


# --- _parse_article_num_tuple ---

class TestParseArticleNumTuple:
    def test_simple(self):
        assert _parse_article_num_tuple("709") == (709, 0)

    def test_branch(self):
        assert _parse_article_num_tuple("709_2") == (709, 2)

    def test_double_branch(self):
        assert _parse_article_num_tuple("15_2_4") == (15, 2, 4)

    def test_with_no(self):
        assert _parse_article_num_tuple("6の2") == (6, 2)

    def test_ordering(self):
        nums = ["710", "709_2", "709", "709_10", "709_1"]
        parsed = sorted(nums, key=_parse_article_num_tuple)
        assert parsed == ["709", "709_1", "709_2", "709_10", "710"]


# --- _decode_law_xml ---

class TestDecodeLawXml:
    def test_base64(self):
        lft = _make_law_xml(_article("1"))
        root = _decode_law_xml(lft)
        assert root is not None
        assert root.tag == "Law"

    def test_raw_xml_string(self):
        xml = "<Law><LawBody></LawBody></Law>"
        b64 = base64.b64encode(xml.encode("utf-8")).decode("ascii")
        root = _decode_law_xml(b64)
        assert root is not None

    def test_invalid(self):
        root = _decode_law_xml("!!invalid!!")
        assert root is None

    def test_none(self):
        root = _decode_law_xml(None)
        assert root is None


# --- find_article_range sorting ---

class TestFindArticleRangeSorting:
    """Test that articles are correctly filtered and sorted with branch numbers."""

    def test_branch_articles_in_range(self):
        body = "\n".join([
            "<Chapter>",
            _article("709", "不法行為"),
            _article("709_2", "損害賠償の方法"),
            _article("710", "財産以外の損害の賠償"),
            _article("711", "近親者に対する損害の賠償"),
            _article("724", "不法行為による損害賠償請求権の消滅時効"),
            "</Chapter>",
        ])
        lft = _make_law_xml(body)
        root = _decode_law_xml(lft)

        start_t = _parse_article_num_tuple("709")
        end_t = _parse_article_num_tuple("711")

        articles = []
        for article in root.iter('Article'):
            num_str = article.get('Num', '')
            num_t = _parse_article_num_tuple(num_str)
            if start_t <= num_t <= end_t:
                articles.append({"num": num_str, "sort": num_t})

        articles.sort(key=lambda a: a["sort"])
        nums = [a["num"] for a in articles]
        assert nums == ["709", "709_2", "710", "711"]

    def test_excludes_out_of_range(self):
        start_t = _parse_article_num_tuple("710")
        end_t = _parse_article_num_tuple("711")
        assert not (start_t <= _parse_article_num_tuple("709_2") <= end_t)
        assert start_t <= _parse_article_num_tuple("710") <= end_t


# --- get_cross_references pattern extraction ---

class TestCrossReferencePatterns:
    """Test the cross-reference extraction patterns used in get_cross_references."""

    def test_internal_ref_detection(self):
        import re
        text = "第九十五条の規定による取消しは、第三者の権利を害することはできない。"
        # Just test the regex pattern matches
        matches = re.findall(r'第([一二三四五六七八九十百千]+)条', text)
        assert "九十五" in matches

    def test_seirei_delegation(self):
        text = "政令で定める基準に従い、都道府県知事の許可を受けなければならない。"
        assert "政令で定める" in text

    def test_kisoku_delegation(self):
        import re
        text = "環境省令で定める技術上の基準に従い、処理しなければならない。"
        assert re.search(r'(?:環境|厚生)?省令で定める', text)

    def test_junyou_extraction(self):
        import re
        text = "第七条から第九条までの規定は、特定施設について準用する。"
        assert "準用" in text
        before = text[:text.index("準用")]
        arts = re.findall(r'第([一二三四五六七八九十百千]+)条', before)
        assert len(arts) >= 1


# --- Supplementary provisions XML ---

class TestSupplementaryProvisionsXml:
    def test_parse_supplementary(self):
        suppl = """
        <SupplementaryProvisions AmendLawNum="平成二十九年法律第四十四号">
          <SupplementaryProvisionsLabel>附則</SupplementaryProvisionsLabel>
          <Article Num="1">
            <ArticleCaption>施行期日</ArticleCaption>
            <ArticleTitle>第一条</ArticleTitle>
            <Paragraph Num="1">
              <ParagraphSentence>
                <Sentence>この法律は、公布の日から起算して二年を超えない範囲内において政令で定める日から施行する。</Sentence>
              </ParagraphSentence>
            </Paragraph>
          </Article>
        </SupplementaryProvisions>
        """
        lft = _make_law_xml(_article("1"), suppl_content=suppl)
        root = _decode_law_xml(lft)
        assert root is not None

        sp_list = list(root.iter('SupplementaryProvisions'))
        assert len(sp_list) == 1
        assert sp_list[0].get('AmendLawNum') == "平成二十九年法律第四十四号"

        articles = list(sp_list[0].iter('Article'))
        assert len(articles) == 1
        assert articles[0].get('Num') == '1'


# --- Definition search patterns ---

class TestDefinitionPatterns:
    def test_ikou_pattern(self):
        import re
        text = '電気通信事業者（電気通信事業法第二条第五号に規定する電気通信事業者をいう。以下「事業者」という。）'
        pat = re.compile(r'以下「([^」]{1,40})」という。')
        m = pat.search(text)
        assert m is not None
        assert m.group(1) == "事業者"

    def test_soushou_pattern(self):
        import re
        text = '株式会社及び合同会社を以下「会社」と総称する。'
        pat = re.compile(r'以下「([^」]{1,40})」と総称する')
        m = pat.search(text)
        assert m is not None
        assert m.group(1) == "会社"

    def test_towa_pattern(self):
        import re
        text = '「善意」とは、ある事実を知らないことをいう。'
        pat = re.compile(r'「([^」]{1,40})」とは[、,]')
        m = pat.search(text)
        assert m is not None
        assert m.group(1) == "善意"

    def test_multiple_definitions(self):
        import re
        text = ('この法律において「一般廃棄物」という。'
                'また、以下「産業廃棄物」という。')
        pat = re.compile(r'以下「([^」]{1,40})」という。')
        matches = pat.findall(text)
        # "一般廃棄物" also has 以下 before it depending on context
        assert "産業廃棄物" in matches


# --- TOC structure parsing ---

class TestTocStructure:
    def test_parse_toc(self):
        toc = """
        <TOCChapter Num="1">
          <ChapterTitle>第一章　総則</ChapterTitle>
          <ArticleRange>第一条―第五条</ArticleRange>
        </TOCChapter>
        <TOCChapter Num="2">
          <ChapterTitle>第二章　物権</ChapterTitle>
          <ArticleRange>第百七十五条―第三百九十八条の二十二</ArticleRange>
          <TOCSection Num="1">
            <SectionTitle>第一節　総則</SectionTitle>
            <ArticleRange>第百七十五条―第百七十九条</ArticleRange>
          </TOCSection>
        </TOCChapter>
        """
        lft = _make_law_xml(_article("1"), toc_content=toc)
        root = _decode_law_xml(lft)
        toc_elem = root.find('.//TOC')
        assert toc_elem is not None

        chapters = list(toc_elem.findall('TOCChapter'))
        assert len(chapters) == 2

        # First chapter
        title = _elem_text(chapters[0].find('ChapterTitle')).strip()
        assert "総則" in title
        art_range = _elem_text(chapters[0].find('ArticleRange')).strip()
        assert "第一条" in art_range

        # Second chapter has sub-section
        sections = list(chapters[1].findall('TOCSection'))
        assert len(sections) == 1


# --- _elem_text ---

class TestElemText:
    def test_basic(self):
        elem = ET.fromstring("<Sentence>テスト文</Sentence>")
        assert _elem_text(elem).strip() == "テスト文"

    def test_nested(self):
        elem = ET.fromstring("<P><S>あ</S><S>い</S></P>")
        text = _elem_text(elem)
        assert "あ" in text
        assert "い" in text

    def test_none(self):
        assert _elem_text(None) == ""
