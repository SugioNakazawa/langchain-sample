# LangChain Simple Server

LangChainを使用してローカルLLM（Ollama）をOpenAI互換APIとして公開するシンプルなサーバーです。MCP統合やエージェント機能を除いた、最小構成のLangChain APIサーバーです。

## 🎯 目的

- **学習用途**: LangChainの基本的な使い方を学ぶ
- **プロトタイピング**: 簡単なLLM統合のテスト
- **軽量サーバー**: 最小限のリソースでLLMを公開

## 🛠️ 機能

- ✅ OpenAI互換のチャット完了API (`/v1/chat/completions`)
- ✅ モデル一覧API (`/v1/models`)
- ✅ Ollamaとの直接統合
- ✅ シンプルな設定とデプロイ
- ❌ MCPサーバー統合（[denchu](/langchain_server/denchu)版で利用可能）
- ❌ エージェント機能（[denchu](/langchain_server/denchu)版で利用可能）

## 📋 必要な環境

- **Python**: 3.8以上
- **Ollama**: ローカルLLMサーバー
- **メモリ**: 最低4GB（モデルにより変動）

## ⚡ クイックスタート

### 1. Ollamaのセットアップ

```bash
# Ollamaのインストール（macOS）
brew install ollama

# モデルのダウンロード
ollama pull qwen3:4b

# Ollamaサーバーの起動
ollama serve
```

### 2. 依存関係のインストール

```bash
# プロジェクトルートから
pip install -r requirements.txt
```

### 3. サーバーの起動

**自動起動（推奨）:**

```bash
# Linux/Mac
./run_langchain_server.sh

# Windows
run_langchain_server.bat
```

**手動起動:**

```bash
cd langchain_server/simple
python langchain_server.py
```

サーバーは `http://localhost:8000` で起動します。

## 🔧 設定

### 環境変数での設定

```bash
# .env.example をコピー
cp .env.example .env

# .env ファイルを編集
OLLAMA_BASE_URL=http://localhost:11434/v1
DEFAULT_MODEL=qwen3:4b
SERVER_PORT=8000
```

### コード内設定

`langchain_server.py` で直接設定を変更：

```python
llm = ChatOpenAI(
    openai_api_base="http://localhost:11434/v1",  # OllamaのURL
    openai_api_key="none",                        # ローカルなので不要
    model="qwen3:4b"                             # 使用するモデル
)
```

## 📡 API使用例

### チャット完了

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "qwen3:4b",
       "messages": [
         {"role": "user", "content": "Pythonでリストをソートする方法を教えて"}
       ]
     }'
```

**レスポンス例:**
```json
{
  "id": "chatcmpl-local",
  "object": "chat.completion",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Pythonでリストをソートするには、組み込みのsort()メソッドやsorted()関数を使用できます。\n\n例：\n```python\n# sort()メソッド（元のリストを変更）\nmy_list = [3, 1, 4, 1, 5]\nmy_list.sort()\nprint(my_list)  # [1, 1, 3, 4, 5]\n\n# sorted()関数（新しいリストを返す）\nmy_list = [3, 1, 4, 1, 5]\nsorted_list = sorted(my_list)\nprint(sorted_list)  # [1, 1, 3, 4, 5]\n```"
      }
    }
  ]
}
```

### モデル一覧

```bash
curl "http://localhost:8000/v1/models"
```

## 🧪 テスト

### 自動テスト実行

```bash
./test_langchain_server.sh
```

### 手動テスト

```bash
# サーバーの起動確認
curl http://localhost:8000/v1/models

# チャット機能テスト
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{"model": "qwen3:4b", "messages": [{"role": "user", "content": "Hello"}]}'
```

## 🔗 他のサーバーとの比較

| 機能 | Simple | Denchu | MCP Blend |
|------|--------|--------|-----------|
| 基本LLM統合 | ✅ | ✅ | ✅ |
| OpenAI互換API | ✅ | ✅ | ✅ |
| MCPサーバー連携 | ❌ | ✅ | ✅ |
| ReActエージェント | ❌ | ✅ | ✅ |
| 外部ツール統合 | ❌ | ✅ | ✅ |
| 電柱チェック機能 | ❌ | ✅ | ❌ |
| 設定の複雑さ | 低 | 中 | 高 |
| リソース使用量 | 低 | 中 | 高 |

## 🐛 トラブルシューティング

### よくある問題

#### 1. Ollamaに接続できない

**症状**: Connection refused エラー
**解決方法**:
```bash
# Ollamaサーバーの起動確認
ollama serve

# プロセス確認
ps aux | grep ollama

# ポート確認
lsof -i :11434
```

#### 2. モデルが見つからない

**症状**: Model not found エラー
**解決方法**:
```bash
# インストール済みモデル確認
ollama list

# モデルのダウンロード
ollama pull qwen3:4b
```

#### 3. ポートが使用中

**症状**: Port 8000 already in use
**解決方法**:
```bash
# ポート使用状況確認
lsof -i :8000

# プロセス終了
kill -9 <PID>

# または別ポート使用
python langchain_server.py --port 8001
```

#### 4. 仮想環境の問題

**解決方法**:
```bash
# 仮想環境の再作成
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r ../../requirements.txt
```

## 📁 ファイル構成

```
simple/
├── langchain_server.py              # メインサーバーファイル
├── run_langchain_server.sh          # Linux/Mac起動スクリプト
├── run_langchain_server.bat         # Windows起動スクリプト
├── test_langchain_server.sh         # テストスクリプト
├── .env.example                     # 設定ファイルの例
└── README.md                        # このファイル
```

## 🚀 応用例

### OpenWebUIとの統合

1. このサーバーを起動（ポート8000）
2. OpenWebUIの設定でAPIエンドポイントを `http://localhost:8000` に設定
3. WebUIからLLMを利用可能

### カスタマイズ例

```python
# 複数モデル対応
models = {
    "qwen3:4b": "qwen3:4b",
    "qwen3:8b": "qwen3:8b",
    "llama2": "llama2:7b"
}

# システムプロンプト追加
system_prompt = "あなたは親切なアシスタントです。"
messages = [SystemMessage(content=system_prompt)] + user_messages
```

## 📈 パフォーマンス

### 推奨システム要件

| モデル | RAM | 応答時間 | 精度 |
|--------|-----|----------|------|
| qwen3:4b | 4GB | 1-3秒 | 良好 |
| qwen3:8b | 6GB | 2-5秒 | 優秀 |
| qwen3:14b | 10GB | 3-8秒 | 最高 |

## 🔗 関連リンク

- **メインプロジェクト**: [langchain-sample](../../README.md)
- **高機能版**: [denchu server](../denchu/README.md)
- **LangChain公式**: https://python.langchain.com/
- **Ollama**: https://ollama.ai/

---

**更新日**: 2025年10月27日