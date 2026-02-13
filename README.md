# e-Gov MCP Server v2 🏛️⚖️

日本政府 [e-Gov 法令API](https://laws.e-gov.go.jp/apidoc/) のための Model Context Protocol (MCP) サーバー。

AIアシスタントやCLIから日本の法令（六法、施行令、施行規則等）を検索・取得できます。

## クイックスタート

### uvx でインストール不要で実行

```bash
uvx e-gov-mcp
```

### Claude Desktop / Cline 設定

```json
{
  "mcpServers": {
    "e-gov-law": {
      "command": "uvx",
      "args": ["e-gov-mcp"]
    }
  }
}
```

### ローカルインストール

```bash
git clone https://github.com/yoshihiro1909/e-gov-mcp.git
cd e-gov-mcp
uv sync
uv run python run_server.py
```

## MCPツール一覧（23個）

### 条文検索・取得

| ツール | 機能 |
|---|---|
| `find_law_article` | 条文検索（XML構造＋項号フィルタ＋本則/附則切替） |
| `batch_find_articles` | 最大200件のバッチ条文検索 |
| `find_article_range` | 条文を範囲指定で取得（枝番条文も正しくソート） |
| `get_law_content` | 法律全文取得（800KB制限対応） |

### 法令検索

| ツール | 機能 |
|---|---|
| `search_laws` | 法律名・種別・番号で検索 |
| `search_laws_by_keyword` | キーワード全文検索 |
| `find_related_laws` | 施行令・施行規則の自動検索 |

### 構造・目次

| ツール | 機能 |
|---|---|
| `get_law_structure` | 法令の目次構造（編・章・節・款・目）を取得 |
| `search_article_captions` | 条文見出し（キャプション）を検索・一覧表示 |
| `get_chapter_articles` | 章・節のタイトル指定で全条文を取得 |
| `get_law_tables` | 別表（別表第一等）を構造化取得 |

### 参照解析

| ツール | 機能 |
|---|---|
| `get_cross_references` | 条文の参照先を軽量抽出（内部参照・施行令・施行規則・準用） |
| `follow_law_chain` | 施行令→施行規則→準用の参照チェーンを追跡 |
| `explain_law_article` | 条文の全相互参照を並列解決（深層モード対応） |
| `bulk_explain_articles` | 複数条文を一括解説（共有参照の重複排除） |
| `reverse_lookup_requirement` | 逆引き検索（ある条文を参照している条文を探す） |
| `search_articles_by_keyword` | 法律内の条文をキーワードで検索 |

### 附則・定義

| ツール | 機能 |
|---|---|
| `get_supplementary_provisions` | 附則を取得（施行日・経過措置・改正附則） |
| `search_definitions` | 定義語の逆引き（「以下○○という」パターン） |

### キャッシュ・メタ

| ツール | 機能 |
|---|---|
| `prefetch_common_laws` | 頻出法律の事前キャッシュ |
| `get_cache_stats` | キャッシュ統計（メモリ＋ディスク） |
| `clear_cache` | キャッシュクリア |
| `get_api_info` | API接続情報 |

## 主な特徴

### XML構造ベース条文抽出
e-Gov APIが返すXMLを`<Article Num="X">`要素で直接パース。正規表現のみの方式と比較してマッチ精度が飛躍的に向上。

### 枝番条文の正確な処理
`第6条の2`、`第12条の3`、`第15条の2の4` 等をタプル比較で正しくソート・範囲指定。

### 条番号の表記ゆれ対応
以下の入力はすべて同じ条文にマッチ：
`6_2` / `6の2` / `6ー2` / `6-2` / `第6条の2` / `６の２`

### 4層キャッシュ
1. **メモリLRU**: 法律検索(2h)、法律内容(1h)、条文(30m)
2. **ディスク**: `.egov_cache/` にJSON保存（30日TTL）
3. **メモリ監視**: psutil統合（オプション）、512MB超で自動クリーンアップ

### 40以上の法令エイリアス
`config/laws.yaml` で管理。略称→正式名称を自動解決：
- 廃掃法 → 廃棄物の処理及び清掃に関する法律
- 道交法 → 道路交通法
- 労基法 → 労働基準法
- 独禁法 → 独占禁止法 等

### 対応法令（ダイレクトマッピング）

**六法:** 憲法、民法、刑法、商法、民事訴訟法、刑事訴訟法

**主要法:** 会社法、労働基準法、所得税法、法人税法、著作権法、特許法、道路交通法、建築基準法、独占禁止法、消費者契約法、フリーランス保護法

**行政法:** 行政手続法、行政事件訴訟法、行政不服審査法、国家賠償法

その他の法令もAPI検索で取得可能。

## CLI

```bash
# 条文検索
uv run python cli.py find_law_article --law 民法 --article 709
uv run python cli.py find_law_article --law 廃棄物処理法 --article 12 --paragraph 5

# 附則を含める
uv run python cli.py find_law_article --law 民法 --article 1 --include-suppl

# 法律検索
uv run python cli.py search_laws --title 著作権法 --limit 5

# キーワード検索
uv run python cli.py search_keyword --keyword 損害賠償 --limit 5

# 関連法令検索
uv run python cli.py find_related_laws --law 廃棄物処理法

# バッチ検索
uv run python cli.py batch --pairs '[{"law":"民法","article":"709"},{"law":"憲法","article":"9"}]'
```

## テスト

```bash
# 全テスト実行
uv run pytest

# カバレッジ付き
uv run pytest --cov=src --cov-report=term-missing

# 特定テスト
uv run pytest tests/test_new_features.py -v
uv run pytest tests/test_coverage.py -v
```

## プロジェクト構成

```
e-gov-mcp/
├── src/
│   ├── mcp_server.py       # MCPサーバー本体（全ツール）
│   └── prompt_loader.py    # プロンプトテンプレート読み込み
├── config/
│   └── laws.yaml           # エイリアス、ダイレクトマッピング、関連法令
├── prompts/
│   └── legal_analysis.md   # 法的分析プロンプト
├── cli.py                  # CLIラッパー
├── run_server.py           # MCPサーバー起動スクリプト
├── tests/                  # pytest テストスイート
├── .egov_cache/            # ディスクキャッシュ（git除外）
└── pyproject.toml
```

## ライセンス

MIT License

## 謝辞

- [ryoooo/e-gov-law-mcp](https://github.com/ryoooo/e-gov-law-mcp) - 本プロジェクトの原型となったリポジトリ
- [e-Gov](https://www.e-gov.go.jp/) - 法令API提供
- [FastMCP](https://github.com/jlowin/fastmcp) - MCPフレームワーク
