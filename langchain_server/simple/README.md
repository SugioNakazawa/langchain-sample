# LangChain Server

LangChainを使用してローカルLLM（Ollama）をOpenAI互換APIとして公開するサーバーです。

## 機能

- OpenAI互換のチャット完了API (`/v1/chat/completions`)
- モデル一覧API (`/v1/models`)
- Ollamaとの統合
- 複数モデルサポート（設定可能）

## 必要な環境

- Python 3.8+
- Ollama（ローカルLLMサーバー）
- 必要なPythonパッケージ（requirements.txtを参照）

## セットアップ

### 1. 仮想環境の作成

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows
```

### 2. 依存関係のインストール

```bash
pip install fastapi uvicorn langchain langchain-openai
```

### 3. Ollamaの起動

```bash
ollama serve
```

利用可能なモデルを確認：
```bash
ollama list
```

## 使用方法

### 自動起動スクリプト

**Linux/Mac:**
```bash
./run_langchain_server.sh
```

**Windows:**
```batch
run_langchain_server.bat
```

### 手動起動

```bash
cd /path/to/openwebui
source venv/bin/activate
python langchain_server/simple/langchain_server.py
```

## API使用例

### チャット完了

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "nautilus-llm",
       "messages": [
         {"role": "user", "content": "こんにちは"}
       ]
     }'
```

### モデル一覧

```bash
curl "http://localhost:8000/v1/models"
```

## テスト

テストスクリプトを実行：

```bash
./test_langchain_server.sh
```

## 設定

設定ファイル `.env.example` をコピーして `.env` として使用できます：

```bash
cp .env.example .env
```

## トラブルシューティング

### Ollamaに接続できない

1. Ollamaが起動していることを確認：
   ```bash
   ollama serve
   ```

2. Ollamaのポートが正しいことを確認（デフォルト: 11434）

3. 利用可能なモデルを確認：
   ```bash
   ollama list
   ```

### ポートが使用中

デフォルトポート8000が使用中の場合、コード内のポート番号を変更してください。

### 仮想環境の問題

仮想環境を再作成：
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ファイル構成

```
simple/
├── langchain_server.py              # メインサーバーファイル
├── run_langchain_server.sh          # Linux/Mac起動スクリプト
├── run_langchain_server.bat         # Windows起動スクリプト
├── test_langchain_server.sh         # テストスクリプト
├── .env.example                     # 設定ファイルの例
└── README.md                        # このファイル
```