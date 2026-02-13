# 使用方法ガイド

## 基本的な使用方法

### 1. サーバーの起動

```bash
# STDIO トランスポート（Claude Desktop等）
uv run python src/mcp_server.py

# SSE トランスポート（Webアプリケーション等）
uv run python src/mcp_server.py --transport sse --port 8000
```

### 2. Claude Desktopでの設定

設定ファイル（`claude_desktop_config.json`）に以下を追加：

```json
{
  "mcpServers": {
    "e-gov-law": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/e-gov-law-mcp",
        "python",
        "src/mcp_server.py"
      ]
    }
  }
}
```

## ツールの詳細

### search_laws - 法令検索

日本の法令・規則を検索します。

**パラメータ:**
- `law_title` (オプション): 法令名での部分一致検索
- `law_type` (オプション): 法令種別での絞り込み
- `limit` (オプション): 取得件数の上限（1-100、デフォルト: 10）
- `offset` (オプション): 開始位置（デフォルト: 0）

**使用例:**
```json
{
  "law_title": "民法",
  "law_type": "Act",
  "limit": 5
}
```

**法令種別の値:**
- `Constitution`: 憲法
- `Act`: 法律
- `CabinetOrder`: 政令
- `ImperialOrder`: 勅令
- `MinisterialOrdinance`: 府省令
- `Rule`: 規則
- `Misc`: その他

### search_laws_by_keyword - キーワード検索

法令本文内でのキーワード全文検索を実行します。

**パラメータ:**
- `keyword` (必須): 検索キーワード
- `law_type` (オプション): 法令種別での絞り込み
- `limit` (オプション): 取得件数の上限（1-20、デフォルト: 5）

**使用例:**
```json
{
  "keyword": "契約",
  "law_type": "Act",
  "limit": 10
}
```

### get_law_content - 法令本文取得

指定した法令の全文内容を取得します。

**パラメータ:**
- `law_id` または `law_num` (どちらか必須): 法令の識別子
- `response_format` (オプション): "json" または "xml"（デフォルト: "json"）

**使用例:**
```json
{
  "law_num": "明治二十九年法律第八十九号",
  "response_format": "json"
}
```

### get_law_revisions - 法令改正履歴

指定した法令の改正履歴を取得します。

**パラメータ:**
- `law_id` または `law_num` (どちらか必須): 法令の識別子

**使用例:**
```json
{
  "law_num": "明治二十九年法律第八十九号"
}
```

## リソース

### api://info

e-Gov法令APIの基本情報と機能一覧を取得できます。

### schema://law_types

利用可能な法令種別とその説明を取得できます。

## エラーハンドリング

### 一般的なエラー

- **入力検証エラー**: パラメータが無効な場合
- **API接続エラー**: e-Gov APIへの接続に失敗した場合
- **HTTPエラー**: APIからエラーレスポンスが返された場合

### エラーレスポンスの例

```
Error: limit must be between 1 and 100
Network Error: Failed to connect to API (ConnectTimeout)
API Error 404: Failed to retrieve law data
```

## 制限事項

1. **レート制限**: e-Gov APIのレート制限に準拠
2. **データ範囲**: e-Gov法令検索で公開されている法令のみ
3. **形式**: JSON/XML形式での取得のみサポート

## トラブルシューティング

### 接続エラー

```bash
# サーバーのログを確認
uv run python src/mcp_server.py --transport sse --port 8000
```

### デバッグモード

環境変数 `PYTHONPATH` と `LOGLEVEL` を設定：

```bash
export LOGLEVEL=DEBUG
uv run python src/mcp_server.py
```

### 依存関係の問題

```bash
# 依存関係を再インストール
uv sync --reinstall
```

## よくある質問

**Q: 法令番号がわからない場合はどうすればよいですか？**
A: まず `search_laws` で法令名を検索し、結果から法令番号を取得してください。

**Q: 大量のデータを取得したい場合は？**
A: `offset` と `limit` パラメータを使用してページネーションを実装してください。

**Q: XML形式とJSON形式の違いは？**
A: JSON形式は構造化されており解析しやすく、XML形式は元の法令構造により近い形式です。

**Q: APIのレート制限はありますか？**
A: e-Gov APIの利用規約に従ってください。過度なリクエストは避けてください。