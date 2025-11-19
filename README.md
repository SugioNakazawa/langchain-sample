# LangChain Sample Project

LangChainを利用したサンプルです。

## 🏗️ システム構成

```
Stand alone type:
LangChain API(Python) →  → LLM (Ollama Port 11434)
          ↓
    MCP Servers
  (Java Tools)

Web type:
User → OpenWebUI (Port 3000) → LangChain API (Port 8000) → LLM (Ollama Port 11434)
                                       ↓
                                  MCP Servers
                                (Java Tools)
```

## 📁 プロジェクト構造

```
langchain-sample/
├── langchain_server/         # LangChainベースのAPIサーバー群
│   ├── xxx/                  # アプリケーションディレクトリ（アプリごと）
│   └── lib/                  # 共有ライブラリ(MCPサーバー)
├── docker/                   # OpenWebUI Docker設定
│   └── openwebui             # OpenWebUI用Docker Compose設定
├── requirements.txt          # Python依存関係
├── logs/                     # システムログ
└── venv/                     # Python仮想環境()
```

## 🚀 クイックスタート

### 1. 環境要件

- **Python**: 3.8以上(推奨 3.13.7)
- **Java**: 11以上 (MCPサーバー用)
- **Docker & Docker Compose**: OpenWebUI用
- **Ollama**: LLMエンジン

### 2. インストール

```bash
# リポジトリのクローン
git clone https://github.com/SugioNakazawa/langchain-sample.git
cd langchain-sample

# Python仮想環境の作成
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate     # Windows

# 依存関係のインストール
pip install -r requirements.txt
```

### 3. Ollamaのセットアップ

```bash
# Ollamaのインストール（macOS）
brew install ollama

# モデルのダウンロード
ollama pull qwen3:14b

# Ollamaサーバーの起動
ollama serve
```

### 4. アプリケーションの起動

langchain_server以下にアプリケーションごとにアプリケーションディレクトリを配置しています。
各アプリケーションはアプリケーションディクトリ内のファイルで実行してください。

| アプリケーション  | 説明                     | 起動PG | UI |
|------------------|--------------------------|---|---|
| chain | LangChainチェーンタイプ | chain.py | console |
| denchu | 電柱チェック業務向けサーバー | app.py | OpenWebUI |
| denchu_auto | 自動化版電柱チェック | app.py | console |
| human-in-the-loop | ヒューマンインザループサーバー | hitl_fastapi_demo.py | Web |
| mcp_blend | MCP統合サンプル | mcp_app_server.py | console |
| rag | RAGサンプル | rag_with_pdf.py | console |
| simple | シンプルなLangChainサーバー | langchain_server.py | OpenWebUI |
| structures_output | 構造化出力サンプル | structured_output.py | console |

## Appendix
### OpenWebUI起動
UI に Web を利用するアプリでは OpenWebUI を起動して使用してください。

```bash
# 別ターミナルでOpenWebUIの起動
cd docker/openwebui
docker-compose up -d
```

アクセス
- OpenWebUI: http://localhost:3000

### Olammaアクセス
- Ollama: http://localhost:11434

---


## 🔗 関連リンク

- **LangChain公式**: https://python.langchain.com/
- **OpenWebUI**: https://github.com/open-webui/open-webui
- **Ollama**: https://ollama.ai/
- **MCP仕様**: https://modelcontextprotocol.io/

---

**Last Updated**: 2025年11月19日
