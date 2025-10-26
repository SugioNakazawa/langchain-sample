# MCP Blend Server

線形プログラミングを使用した油脂ブレンド最適化システムです。MCPサーバーとLangChainを統合したアプリケーションサーバーを提供します。

## 機能

- **MCPサーバー**: 線形プログラミングによる油脂ブレンド最適化
- **Appサーバー**: OpenAI互換APIとLangChainエージェントの統合
- **最適化アルゴリズム**: PuLPライブラリを使用したコスト最小化

## 必要な環境

- Python 3.8+
- Ollama（ローカルLLMサーバー）
- 必要なPythonパッケージ：
  - fastapi
  - uvicorn
  - langchain
  - langchain-openai
  - pulp
  - requests

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
pip install fastapi uvicorn langchain langchain-openai pulp requests
```

### 3. Ollamaの起動

```bash
ollama serve
```

## 使用方法

### 自動起動スクリプト

**Linux/Mac:**
```bash
./run_mcp_blend_server.sh
```

**Windows:**
```batch
run_mcp_blend_server.bat
```

### 起動オプション

スクリプト実行時に以下の選択肢が表示されます：

1. **MCPサーバーのみ起動** (ポート 9100)
2. **Appサーバーのみ起動** (ポート 8000)
3. **両方のサーバーを起動**
4. **テスト実行**

### 手動起動

```bash
# MCPサーバー
python langchain_server/mcp_blend/mcp_blend_server.py

# Appサーバー（別ターミナル）
python langchain_server/mcp_blend/mcp_app_server.py
```

## API使用例

### MCPサーバー（直接呼び出し）

```bash
curl -X POST "http://localhost:9100" \
     -H "Content-Type: application/json" \
     -d '{
       "tool_id": "optimize_blend",
       "input": {
         "oils": [
           {"name": "Soybean", "cost": 1.1, "iodine": 120},
           {"name": "Palm", "cost": 0.9, "iodine": 80},
           {"name": "Rapeseed", "cost": 1.0, "iodine": 110}
         ],
         "demand": 1000
       }
     }'
```

### Appサーバー（OpenAI互換API）

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "nautilus-llm",
       "messages": [
         {"role": "user", "content": "最適ブレンドを計算してください"}
       ]
     }'
```

## テスト

テストスクリプトを実行：

```bash
./test_mcp_blend_server.sh
```

## 最適化アルゴリズム

### 目的関数
コストの最小化：
```
minimize: Σ(cost_i × quantity_i)
```

### 制約条件
1. **需要制約**: `Σ quantity_i = demand`
2. **品質制約**: `Σ(iodine_i × quantity_i) / demand ≥ min_iodine`
3. **非負制約**: `quantity_i ≥ 0`

### 入力パラメータ
- `oils`: 油脂リスト（名前、コスト、ヨウ素価）
- `demand`: 総需要量
- `min_iodine`: 最小ヨウ素価（デフォルト: 100）

### 出力
- `blend`: 各油脂の最適配合量
- `total_cost`: 総コスト

## アーキテクチャ

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ユーザー      │───▶│  Appサーバー    │───▶│   Ollama        │
│                 │    │  (port: 8000)   │    │  (port: 11434)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                │ HTTP Request
                                ▼
                       ┌─────────────────┐
                       │  MCPサーバー    │
                       │  (port: 9100)   │
                       │  [最適化エンジン]│
                       └─────────────────┘
```

## トラブルシューティング

### MCPサーバーに接続できない

1. MCPサーバーが起動していることを確認
2. ポート9100が利用可能であることを確認
3. ファイアウォール設定を確認

### Ollamaに接続できない

1. Ollamaが起動していることを確認：
   ```bash
   ollama serve
   ```

2. 利用可能なモデルを確認：
   ```bash
   ollama list
   ```

### 最適化エラー

1. 入力データの形式を確認
2. 制約条件が満たせない場合のエラー
3. PuLPライブラリのインストールを確認

## ファイル構成

```
mcp_blend/
├── mcp_blend_server.py          # MCPサーバー（最適化エンジン）
├── mcp_app_server.py            # Appサーバー（API統合）
├── run_mcp_blend_server.sh      # Linux/Mac起動スクリプト
├── run_mcp_blend_server.bat     # Windows起動スクリプト
├── test_mcp_blend_server.sh     # テストスクリプト
└── README.md                    # このファイル
```