# 開発ガイド

## 開発環境のセットアップ

### 前提条件

- Python 3.9以上
- [uv](https://docs.astral.sh/uv/) パッケージマネージャー

### セットアップ手順

```bash
# リポジトリをクローン
git clone https://github.com/your-username/e-gov-law-mcp.git
cd e-gov-law-mcp

# 開発依存関係を含む全ての依存関係をインストール
uv sync --dev

# pre-commitフックをインストール（推奨）
uv run pre-commit install
```

## プロジェクト構造

```
e-gov-law-mcp/
├── src/
│   └── mcp_server.py      # メインのMCPサーバー実装
├── examples/              # 使用例
│   ├── basic_search.py
│   ├── keyword_search.py
│   ├── law_content.py
│   └── resources_demo.py
├── tests/                 # テストファイル
│   └── test_mcp_server.py
├── docs/                  # ドキュメント
│   ├── USAGE.md
│   └── DEVELOPMENT.md
├── pyproject.toml         # プロジェクト設定
├── README.md
└── LICENSE
```

## 開発ワークフロー

### 1. コード作成

```bash
# 新しい機能ブランチを作成
git checkout -b feature/new-feature

# コードを編集
# ...

# コードフォーマットを実行
uv run black src/ tests/ examples/

# リント検査を実行
uv run ruff check src/ tests/ examples/

# 型チェックを実行
uv run mypy src/
```

### 2. テスト実行

```bash
# 全テストを実行
uv run pytest

# カバレッジレポート付きで実行
uv run pytest --cov=src --cov-report=html

# 特定のテストファイルのみ実行
uv run pytest tests/test_mcp_server.py

# テストを詳細モードで実行
uv run pytest -v
```

### 3. 品質チェック

```bash
# 全ての品質チェックを実行
uv run pre-commit run --all-files

# または個別に実行
uv run black --check src/ tests/ examples/
uv run ruff check src/ tests/ examples/
uv run mypy src/
```

## 新機能の追加

### 1. 新しいツールの追加

```python
@mcp.tool
async def new_tool(param1: str, param2: int = 10) -> str:
    """
    新しいツールの説明
    
    Args:
        param1: パラメータ1の説明
        param2: パラメータ2の説明（デフォルト: 10）
    
    Returns:
        結果の説明
    """
    # 入力検証
    if not param1:
        return "Error: param1 is required"
    
    # 実装
    async with await get_http_client() as client:
        try:
            # API呼び出し
            response = await client.get("/new-endpoint", params={"param1": param1})
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"New tool error: {e}")
            return f"Error: {str(e)}"
```

### 2. 新しいリソースの追加

```python
@mcp.resource("schema://new_resource")
def get_new_resource() -> dict:
    """新しいリソースの説明"""
    return {
        "data": "リソースデータ",
        "metadata": {
            "version": "1.0",
            "description": "説明"
        }
    }
```

### 3. テストの追加

```python
class TestNewFeature:
    @pytest.mark.asyncio
    async def test_new_tool_success(self):
        """新しいツールの成功ケースをテスト"""
        # テスト実装
        pass
    
    @pytest.mark.asyncio
    async def test_new_tool_validation(self):
        """新しいツールの入力検証をテスト"""
        # テスト実装
        pass
```

## デバッグ

### ログレベルの設定

```bash
# デバッグモードでサーバーを起動
export PYTHONLOGLEVEL=DEBUG
uv run python src/mcp_server.py
```

### MCPクライアントでのテスト

```python
# デバッグ用のクライアントスクリプト
import asyncio
from fastmcp import Client

async def debug_test():
    client = Client(["uv", "run", "python", "src/mcp_server.py"])
    
    async with client:
        # ツール一覧を取得
        tools = await client.list_tools()
        print("Available tools:")
        for tool in tools:
            print(f"  - {tool.name}")
        
        # リソース一覧を取得
        resources = await client.list_resources()
        print("Available resources:")
        for resource in resources:
            print(f"  - {resource.uri}")

if __name__ == "__main__":
    asyncio.run(debug_test())
```

## パフォーマンス最適化

### 1. HTTPクライアントの最適化

```python
# 接続プールの設定
async def get_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=API_URL,
        timeout=30.0,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
    )
```

### 2. キャッシュの実装

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_resource():
    """キャッシュされたリソース"""
    return expensive_operation()
```

## セキュリティ

### 1. 入力検証

```python
def validate_input(value: str, max_length: int = 100) -> bool:
    """入力値を検証"""
    if not value or len(value) > max_length:
        return False
    # その他の検証ロジック
    return True
```

### 2. ログのサニタイズ

```python
import logging

# 機密情報をログに出力しない
logger.info(f"Processing request for user: {user_id[:8]}...")
```

## リリース準備

### 1. バージョン更新

```bash
# pyproject.tomlのバージョンを更新
# version = "1.1.0"
```

### 2. CHANGELOG作成

```markdown
## [1.1.0] - 2024-XX-XX

### Added
- 新機能の説明

### Changed
- 変更された機能の説明

### Fixed
- 修正されたバグの説明
```

### 3. リリースの作成

```bash
# タグを作成
git tag v1.1.0

# GitHubにプッシュ
git push origin v1.1.0
```

## 貢献ガイドライン

### 1. プルリクエストの作成

1. Issueを作成（バグ報告や機能要求）
2. 機能ブランチを作成
3. 変更を実装
4. テストを追加/更新
5. ドキュメントを更新
6. プルリクエストを作成

### 2. コーディング規約

- **言語**: 日本語のコメントとドキュメント
- **フォーマット**: Black、Ruffを使用
- **型ヒント**: 必須
- **テスト**: 新機能には必ずテストを追加

### 3. コミットメッセージ

```
feat: 新機能の追加

詳細な説明...

Closes #123
```

## トラブルシューティング

### よくある問題

1. **依存関係の競合**
   ```bash
   uv sync --reinstall
   ```

2. **テストの失敗**
   ```bash
   uv run pytest -v --tb=short
   ```

3. **型エラー**
   ```bash
   uv run mypy src/ --show-error-codes
   ```

### サポート

- GitHub Issues: バグレポートや機能要求
- GitHub Discussions: 質問や議論
- 開発者向けドキュメント: このファイル