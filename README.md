# LangChain Sample Project

LangChainとMCP (Model Context Protocol) サーバーを使用したAIアシスタントのサンプルプロジェクトです。OpenWebUIと連携して、電柱チェック業務などの実用的なタスクを自動化できます。

## 🏗️ システム構成

```
User → OpenWebUI (Port 3000) → LangChain API (Port 8000) → LLM (Ollama Port 11434)
                                       ↓
                                  MCP Servers
                                (Java Tools)
```

## 📁 プロジェクト構造

```
langchain-sample/
├── langchain_server/           # LangChainベースのAPIサーバー群
│   ├── denchu/                # 電柱チェック業務向けサーバー
│   │   ├── app.py            # メインAPIサーバー
│   │   ├── app.json          # 設定ファイル
│   │   └── logs/             # ログディレクトリ
│   ├── denchu_auto/          # 自動化版電柱チェックサーバー
│   ├── lib/                  # 共有ライブラリ
│   ├── mcp_blend/            # MCP統合サンプル
│   └── simple/               # シンプルなLangChainサーバー
├── openwebui-docker/          # OpenWebUI Docker設定
│   ├── docker-compose.yaml
│   └── README.md
├── requirements.txt           # Python依存関係
├── logs/                     # システムログ
└── venv/                     # Python仮想環境
```

## 🚀 クイックスタート

### 1. 環境要件

- **Python**: 3.8以上
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

```bash
# 電柱チェックサーバーの起動
cd langchain_server/denchu
python app.py

# 別ターミナルでOpenWebUIの起動
cd openwebui-docker
docker-compose up -d
```

### 5. アクセス

- **OpenWebUI**: http://localhost:3000
- **LangChain API**: http://localhost:8000
- **Ollama**: http://localhost:11434

## ⚙️ 設定ファイル

### app.json の主要設定

```json
{
    "openai_api_base": "http://localhost:11434/v1",
    "agent_model": "qwen3:14b",
    "log_folder": "logs",
    "report_folder": "out",
    "system_prompts": [
        "あなたはMCPサーバーを使用するAIアシスタントです。",
        "MCP Toolの結果を優先して回答として採用してください。",
        "チェック項目のOK/NGの更新指示があった場合はMCP Toolを使って更新してください。"
    ],
    "color": {
        "tool": "32",
        "system": "36", 
        "agent": "93",
        "user": "91"
    },
    "mcp_servers": {
        "checklist": {
            "transport": "stdio",
            "command": "java",
            "args": [
                "-Xms64m",
                "-Xmx128m", 
                "-jar",
                "./sandbox-mcp-checklist-0.0.1-SNAPSHOT-all.jar",
                "--type", "3",
                "-c", "tcp://localhost:12345"
            ]
        }
    }
}
```

## 🔗 API仕様

### チャット補完エンドポイント

```http
POST /v1/chat/completions
Content-Type: application/json

{
  "messages": [
    {
      "role": "user", 
      "content": "電柱567のチェックを開始して"
    }
  ]
}
```

**レスポンス例:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1698765432,
  "model": "qwen3:14b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "電柱567のチェック作業を開始します。担当IDを教えてください。"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 25,
    "total_tokens": 37
  }
}
```

### モデル一覧

```http
GET /v1/models
```

## 🛠️ 主要機能

### 1. ReActエージェント
- **推論-行動サイクル**: 問題を分析し、適切なツールを選択
- **動的ツール選択**: 状況に応じて最適な処理を実行
- **メモリ機能**: 会話履歴を保持した対話

### 2. MCP統合
- **外部ツール連携**: Java製MCPサーバーとの非同期通信
- **複数サーバー対応**: 同時に複数のMCPサーバーを利用
- **エラーハンドリング**: 接続失敗時の適切なフォールバック

### 3. OpenWebUI統合
- **OpenAI互換API**: 既存のUIをそのまま利用
- **ストリーミング対応**: リアルタイム応答表示
- **マルチモーダル**: 将来的な音声・画像対応

### 4. ログ・監視機能
- **カラー付きログ**: 種別ごとの色分け表示
- **ファイル出力**: 永続化されたログ保存
- **実行時間計測**: パフォーマンス監視

## 📋 使用例

### 電柱チェック業務

```
👤 User: "担当ID 2、電柱567でチェック開始"
🤖 AI: "担当ID 2、電柱ID 567でチェック作業を開始します"

👤 User: "項目1はOK"  
🤖 AI: "項目1をOKとして記録しました"

👤 User: "項目10は古くなってるのでNG"
🤖 AI: "項目10をNGとして記録し、備考に「古い」を追加しました"

👤 User: "レポートを作成して"
🤖 AI: "チェック結果レポートを作成しました。OK: 9項目、NG: 1項目"
```

## 🐛 トラブルシューティング

### よくある問題

#### 1. `'dict' object has no attribute 'find'`
**原因**: OpenWebUI連携時のレスポンス形式エラー
**解決**: レスポンスのcontentフィールドが文字列になっているか確認

#### 2. `RuntimeWarning: coroutine was never awaited`
**原因**: 非同期関数の呼び出しエラー
**解決**: `await`キーワードを適切に使用

#### 3. `MultiServerMCPClient object has no attribute 'connect'`
**原因**: MCPクライアントのAPI使用方法エラー
**解決**: `connect()`メソッドの呼び出しを削除

#### 4. モデルが応答しない
**原因**: Ollamaサーバーの未起動またはモデル未インストール
**解決**: 
```bash
ollama serve
ollama pull qwen3:14b
```

#### 5. MCPサーバー接続エラー
**原因**: JARファイルの不在またはパス間違い
**解決**: 
- JARファイルの存在確認
- app.jsonのパス設定確認
- Javaバージョン確認

### ログレベル設定

```python
# デバッグモード有効化
logging.basicConfig(level=logging.DEBUG)

# 特定ログャーの調整
logging.getLogger('langchain').setLevel(logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
```

## 🔧 開発者向け情報

### 主要依存関係

- **LangChain**: 1.0.2 - エージェントフレームワーク
- **LangGraph**: 1.0.1 - グラフベースワークフロー
- **FastAPI**: 0.120.0 - REST API フレームワーク
- **langchain-mcp-adapters**: 0.1.11 - MCP統合
- **langchain-openai**: 1.0.1 - OpenAI互換LLM

### アーキテクチャ設計

```python
# 非同期初期化パターン
async def initialize_components():
    global agent, tools, llm
    
    # MCPクライアント初期化
    client = MultiServerMCPClient(mcp_config)
    tools = await client.get_tools()
    
    # ReActエージェント構築
    agent = create_react_agent(
        llm=llm,
        tools=tools, 
        state_modifier=system_message
    )
```

### カスタマイズポイント

1. **システムプロンプト**: `app.json`の`system_prompts`配列
2. **LLMモデル**: `agent_model`設定
3. **MCPサーバー**: `mcp_servers`設定
4. **ログ設定**: `log_folder`と色設定

## 📊 パフォーマンス

### 推奨システム要件

- **CPU**: 4コア以上
- **メモリ**: 8GB以上（LLMモデルにより増加）
- **ストレージ**: 10GB以上の空き容量
- **ネットワーク**: ブロードバンド接続

### ベンチマーク例

| モデル | 応答時間 | メモリ使用量 | 精度 |
|--------|----------|--------------|------|
| qwen3:8b | 2-5秒 | 6GB | 良好 |
| qwen3:14b | 3-8秒 | 10GB | 優秀 |
| qwen3:32b | 5-15秒 | 20GB | 最高 |

## 🤝 コントリビューション

### 開発の流れ

1. **Issue作成**: バグ報告や機能要望
2. **Fork & Branch**: 機能ブランチの作成
3. **実装**: コード変更とテスト
4. **Pull Request**: レビュー依頼
5. **マージ**: 承認後の統合

### コーディング規約

- **Python**: PEP 8準拠
- **コミット**: [Conventional Commits](https://conventionalcommits.org/)
- **ドキュメント**: docstring必須
- **テスト**: pytest使用

## 📄 ライセンス

このプロジェクトは[MIT License](LICENSE)の下で公開されています。

## 🔗 関連リンク

- **LangChain公式**: https://python.langchain.com/
- **OpenWebUI**: https://github.com/open-webui/open-webui
- **Ollama**: https://ollama.ai/
- **MCP仕様**: https://modelcontextprotocol.io/

## 📞 サポート

- **Issues**: GitHubのIssueページ
- **Discussions**: プロジェクトディスカッション
- **Wiki**: 詳細ドキュメント

---

**Last Updated**: 2025年10月27日
