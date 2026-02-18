# e-gov-mcp

日本の法令を検索・取得・解析するMCPサーバー。[e-Gov 法令API](https://laws.e-gov.go.jp/apidoc/) を利用。

## できること

- **条文検索・取得** — 「民法709条」のように指定して条文を取得。枝番（第6条の2）や範囲指定にも対応
- **参照チェーン追跡** — 条文内の「施行令第○条」「準用する」等の参照先を自動解決し、芋づる式に辿る
- **キーワード全文検索** — 法律名だけでなく条文内容からも検索
- **構造把握** — 編・章・節・款・目の目次構造、条文見出し一覧、別表の取得
- **附則・定義語** — 施行日・経過措置の取得、「以下○○という」の逆引き
- **バッチ処理** — 最大200件の条文を一括取得

## インストール

```bash
# uvx で直接実行（インストール不要）
uvx e-gov-mcp

# またはローカルで
git clone https://github.com/kuro6061/e-gov-mcp.git
cd e-gov-mcp
uv sync
uv run python run_server.py
```

## MCP設定

Claude Desktop / Cline:

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

## ツール一覧（23個）

### 条文の検索・取得

- **`find_law_article`** — 法律名＋条番号で条文取得（項・号フィルタ、本則/附則切替）
- **`batch_find_articles`** — 最大200件の一括条文取得
- **`find_article_range`** — 条文を範囲で取得（枝番も正しくソート）
- **`get_law_content`** — 法律全文取得（800KB超の大規模法にも対応）
- **`get_chapter_articles`** — 章・節のタイトル指定で全条文取得

### 法令の検索

- **`search_laws`** — 法律名・種別・番号で検索
- **`search_laws_by_keyword`** — キーワード全文検索
- **`find_related_laws`** — 施行令・施行規則を自動検索
- **`search_article_captions`** — 条文見出し（キャプション）の検索・一覧
- **`search_articles_by_keyword`** — 法律内の条文をキーワード検索

### 構造・参照解析

- **`get_law_structure`** — 法令の目次構造（編→章→節→款→目）
- **`get_law_tables`** — 別表を構造化取得
- **`get_cross_references`** — 条文の参照先を抽出（内部参照・施行令・準用）
- **`follow_law_chain`** — 施行令→施行規則→準用の参照チェーンを追跡
- **`explain_law_article`** — 条文の全相互参照を並列解決（深層モード対応）
- **`bulk_explain_articles`** — 複数条文を一括解説（共有参照の重複排除）
- **`reverse_lookup_requirement`** — ある条文を参照している条文を逆引き

### 附則・定義

- **`get_supplementary_provisions`** — 附則（施行日・経過措置・改正附則）
- **`search_definitions`** — 定義語の逆引き（「以下○○という」パターン）

### キャッシュ

- **`prefetch_common_laws`** — 頻出法律の事前キャッシュ
- **`get_cache_stats`** — キャッシュ統計
- **`clear_cache`** — キャッシュクリア
- **`get_api_info`** — API接続情報

## 条番号の表記ゆれ

以下はすべて同じ条文にマッチする:

`6_2` / `6の2` / `6ー2` / `6-2` / `第6条の2` / `６の２`

## 法令エイリアス

`config/laws.yaml` で40以上の略称を定義:

- 廃掃法 → 廃棄物の処理及び清掃に関する法律
- 道交法 → 道路交通法
- 労基法 → 労働基準法
- 独禁法 → 独占禁止法 等

六法（憲法・民法・刑法・商法・民訴法・刑訴法）と主要法はダイレクトマッピングで即座にアクセス可能。

## CLI

```bash
# 条文取得
uv run python cli.py find_law_article --law 民法 --article 709
uv run python cli.py find_law_article --law 廃棄物処理法 --article 12 --paragraph 5

# 法令検索
uv run python cli.py search_laws --title 著作権法

# キーワード検索
uv run python cli.py search_keyword --keyword 損害賠償 --limit 5

# 関連法令
uv run python cli.py find_related_laws --law 廃棄物処理法

# バッチ
uv run python cli.py batch --pairs '[{"law":"民法","article":"709"},{"law":"憲法","article":"9"}]'
```

## テスト

```bash
uv run pytest
uv run pytest --cov=src --cov-report=term-missing
```

## プロジェクト構成

```
e-gov-mcp/
├── src/
│   ├── mcp_server.py       # MCPサーバー本体
│   └── prompt_loader.py    # プロンプトテンプレート
├── config/
│   └── laws.yaml           # エイリアス・ダイレクトマッピング
├── prompts/
│   └── legal_analysis.md   # 法的分析プロンプト
├── cli.py                  # CLI
├── run_server.py           # サーバー起動
└── tests/                  # テスト
```

## 謝辞

- [ryoooo/e-gov-law-mcp](https://github.com/ryoooo/e-gov-law-mcp) — フォーク元
- [e-Gov 法令API](https://laws.e-gov.go.jp/apidoc/)
- [FastMCP](https://github.com/jlowin/fastmcp)

## ライセンス

MIT
