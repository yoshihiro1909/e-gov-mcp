# WSL環境でのトラブルシューティング

## よくあるエラーと解決策

### 1. "uv: command not found" エラー

**症状：** WSL経由でuvコマンドが見つからない

**解決策：**

#### 方法1: フルパスでuvを指定
```json
{
  "mcpServers": {
    "e-gov-law": {
      "command": "wsl",
      "args": [
        "bash",
        "-c",
        "cd /home/username/dev/e-gov-law-mcp && /home/username/.local/bin/uv run python src/mcp_server.py"
      ]
    }
  }
}
```

#### 方法2: 仮想環境を直接使用
```json
{
  "mcpServers": {
    "e-gov-law": {
      "command": "wsl",
      "args": [
        "bash",
        "-c",
        "cd /home/username/dev/e-gov-law-mcp && source .venv/bin/activate && python src/mcp_server.py"
      ]
    }
  }
}
```

### 2. "ModuleNotFoundError: No module named 'fastmcp'" エラー

**症状：** Pythonモジュールが見つからない

**解決策：**

WSL内で依存関係を再インストール：
```bash
# WSL内で実行
cd /home/username/dev/e-gov-law-mcp
uv sync
```

### 3. PATHの問題

**症状：** WSLのログインシェルでPATHが正しく設定されない

**解決策：**

`.bashrc`または`.profile`でPATHを設定：
```bash
# ~/.bashrcに追加
export PATH="$HOME/.local/bin:$PATH"
```

### 4. 権限エラー

**症状：** Permission deniedエラー

**解決策：**

スクリプトに実行権限を付与：
```bash
# WSL内で実行
chmod +x /home/username/dev/e-gov-law-mcp/src/mcp_server.py
```

### 5. WSLディストリビューションの指定

複数のWSLディストリビューションがある場合：

```json
{
  "mcpServers": {
    "e-gov-law": {
      "command": "wsl",
      "args": [
        "-d",
        "Ubuntu",  // ディストリビューション名を指定
        "bash",
        "-c",
        "cd /home/username/dev/e-gov-law-mcp && uv run python src/mcp_server.py"
      ]
    }
  }
}
```

## デバッグ方法

### 1. WSL内で直接テスト

まず、WSL内で直接動作確認：
```bash
# WSL内で実行
cd /home/username/dev/e-gov-law-mcp
uv run python src/mcp_server.py
```

### 2. Claude Desktopのログ確認

ログファイルの場所：
```
%APPDATA%\\Claude\\logs\\
```

### 3. 詳細なエラー出力

デバッグ用の設定：
```json
{
  "mcpServers": {
    "e-gov-law": {
      "command": "wsl",
      "args": [
        "bash",
        "-c",
        "cd /home/username/dev/e-gov-law-mcp && uv run python src/mcp_server.py 2>&1 | tee /tmp/mcp-debug.log"
      ]
    }
  }
}
```

その後、WSL内でログを確認：
```bash
cat /tmp/mcp-debug.log
```

## 推奨される設定

最も安定した設定：

```json
{
  "mcpServers": {
    "e-gov-law": {
      "command": "wsl",
      "args": [
        "bash",
        "-c",
        "cd /home/username/dev/e-gov-law-mcp && source .venv/bin/activate && python src/mcp_server.py"
      ],
      "env": {
        "EGOV_API_URL": "https://laws.e-gov.go.jp/api/2",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

`PYTHONUNBUFFERED=1`を追加することで、出力がバッファリングされずに即座に表示されます。