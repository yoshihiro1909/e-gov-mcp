# Windows Setup Guide for e-Gov Law MCP Server

## 🚨 Current Issue Analysis

**重要な知識**: Windows環境では `uvx fastmcp install` コマンドが動作しません。

Claude DesktopがFastMCPコマンドを使用しているため、以下のエラーが発生します：

```
ERROR Failed to run: No module named 'yaml'
```

これは、FastMCPの分離された環境が、インストール済みの依存関係にアクセスできないためです。

## ✅ Quick Fix Steps (動作確認済み)

### 1. Update Claude Desktop Configuration

**Location**: `%APPDATA%\Claude\claude_desktop_config.json`
**Full Path**: `C:\Users\ryoki\AppData\Roaming\Claude\claude_desktop_config.json`

**⚠️ 重要**: Windows環境では、必ず `run_server.py` を使用してください。

**Replace your current configuration with**:
```json
{
  "mcpServers": {
    "e-gov-law": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "C:\\Users\\ryoki\\Claude\\mcp\\e-gov-law-mcp",
        "python",
        "run_server.py"
      ]
    }
  }
}
```

> **💡 ヒント**: `run_server.py` は依存関係チェック機能が組み込まれており、"No module named 'yaml'"エラーを防止します。

### 2. Install Dependencies in Your Project Directory

Open Command Prompt or PowerShell in `C:\Users\ryoki\Claude\mcp\e-gov-law-mcp` and run:

```cmd
# Method 1: Using uv (recommended)
uv sync

# Method 2: If uv sync fails, use pip directly
pip install fastmcp>=2.8.1 httpx>=0.24.0 PyYAML>=6.0

# Optional: For performance monitoring
pip install psutil
```

### 3. Verify run_server.py Exists

Make sure `run_server.py` exists in your project root. If not, pull the latest changes:

```cmd
cd C:\Users\ryoki\Claude\mcp\e-gov-law-mcp
git pull origin main
```

### 4. Test the Configuration

Before restarting Claude Desktop, test that the server works:

```cmd
cd C:\Users\ryoki\Claude\mcp\e-gov-law-mcp
uv run python run_server.py
```

You should see initialization messages without the "No module named 'yaml'" error.

## 🔧 Alternative Solutions

### Manual pip installation

If uv is causing issues:

```cmd
cd C:\Users\ryoki\Claude\mcp\e-gov-law-mcp
pip install fastmcp httpx PyYAML psutil
```

Then use this simpler configuration:
```json
{
  "mcpServers": {
    "e-gov-law": {
      "command": "python",
      "args": [
        "C:\\Users\\ryoki\\Claude\\mcp\\e-gov-law-mcp\\run_server.py"
      ],
      "cwd": "C:\\Users\\ryoki\\Claude\\mcp\\e-gov-law-mcp"
    }
  }
}
```

## 🎯 Expected Success Log

After correct configuration, you should see:
```
[info] Server started and connected successfully
[info] Message from client: {"method":"initialize"...}
```

Without any "No module named" errors.

## 📝 重要なポイント

1. **Windowsでは `uvx fastmcp install` は使用できません**
2. **必ず `run_server.py` を使用**
3. **絶対パスで設定**
4. **バックスラッシュをエスケープ** (`\\`)
5. **必要に応じて `uv sync` を実行**

## 📝 トラブルシューティング順序

1. `uv sync` で依存関係をインストール
2. Claude Desktop設定で `run_server.py` を使用
3. 絶対パスでプロジェクトディレクトリを指定
4. Claude Desktopを再起動
5. サーバーが正常に起動することを確認