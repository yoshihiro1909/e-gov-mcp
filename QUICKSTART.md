# e-Gov Law MCP Server クイックスタートガイド

## 🚨 重要：OS別セットアップ

### 📋 Linux/macOS環境

```bash
# 1. リポジトリをクローン
git clone https://github.com/ryoooo/e-gov-law-mcp.git
cd e-gov-law-mcp

# 2. FastMCP CLIで設定
uvx fastmcp install src/mcp_server.py:mcp -n "e-Gov Law Server"
```

### 🪟 Windows環境（⚠️ FastMCPインストール非対応）

**Windows環境ではFastMCPのインストールコマンドは動作しません。** 手動設定が必要です。

```bash
# 1. リポジトリをクローン  
git clone https://github.com/ryoooo/e-gov-law-mcp.git
cd e-gov-law-mcp

# 2. 依存関係をインストール
uv sync

# 3. Claude Desktop設定を手動で追加
# %APPDATA%\Claude\claude_desktop_config.json
```

**Windows用設定内容**:
```json
{
  "mcpServers": {
    "e-gov-law": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "C:\\Users\\[ユーザー名]\\path\\to\\e-gov-law-mcp",
        "python",
        "run_server.py"
      ]
    }
  }
}
```

> **💡 ポイント**: Windows環境では `run_server.py` を使用することで、依存関係チェック機能が動作します。

## 使い方

Claude Desktopを再起動後、以下のように質問してください：

```
民法192条を教えて
憲法9条について説明して
会社法325条の3を調べて
労働基準法の有給休暇について検索して
```

## トラブルシューティング

### uvxがインストールされていない場合

```bash
# uvをインストール（uvxも含まれます）
curl -LsSf https://astral.sh/uv/install.sh | sh

# または
pip install uv
```

### Windows環境でuvがインストールされていない場合

PowerShellで実行：
```powershell
# uvのインストール
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# ⚠️ 注意: Windows環境では uvx fastmcp install は動作しません
# 上記の手動設定方法を使用してください
```

### 📝 詳細な設定方法

- **Windowsユーザー**: [WINDOWS_SETUP.md](WINDOWS_SETUP.md) を参照
- **一般的な設定**: [README.md](README.md) を参照
- **トラブルシューティング**: [WSL_TROUBLESHOOTING.md](WSL_TROUBLESHOOTING.md) を参照