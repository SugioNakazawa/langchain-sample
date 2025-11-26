# HITL (Human-in-the-Loop) LLM Demo

LLMを使用したHuman-in-the-Loop（人間参加型）システムのデモアプリケーションです。

## 概要

このアプリケーションは、LLMの生成結果に対して人間がレビュー・承認を行うワークフローを実装しています：

1. **LLM生成**: ユーザーのプロンプトに対してLLMが回答を生成
2. **自己評価**: LLMが自身の回答の信頼度（0.0〜1.0）を評価
3. **条件分岐**: 
   - 信頼度が高い（≧0.8）→ 自動承認・公開
   - 信頼度が低い（<0.8）→ 人間レビュー待ちキューに追加
4. **人間レビュー**: レビュアーが承認・編集・却下を選択
5. **公開**: 承認された回答を保存・表示

## 機能

### 🤖 自動生成
- Ollama（ローカルLLM）による回答生成
- モデル: `qwen3:4b`（変更可能）

### 🔍 自己評価
- LLM自身が回答の信頼度を評価
- 評価基準: 0.0（低）〜 1.0（高）
- 閾値: 0.8（調整可能）

### 👤 人間レビュー
- **Approve**: そのまま承認して公開
- **Edit**: 回答を編集してから公開
- **Reject**: 却下（削除）

### 📊 管理画面
- レビュー待ちアイテムの一覧
- 公開済みアイテムの一覧
- シンプルなWeb UI

## セットアップ

### 1. 必要なパッケージをインストール

```bash
# 仮想環境を有効化
cd /Users/sugionakazawa/github/langchain-sample
source venv/bin/activate

# パッケージをインストール
pip install flask ollama tinydb
```

### 2. Ollamaモデルの準備

```bash
# qwen3:4bモデルをダウンロード
ollama pull qwen3:4b

# Ollamaサーバーが起動していることを確認
ollama list
```

### 3. アプリケーションの起動

```bash
cd langchain_server/hitl_llm
source ../../venv/bin/activate
python hitl_llm.py
```

### 4. ブラウザでアクセス

```
http://localhost:5001
```

## 使い方

### 基本的なワークフロー

1. **トップページ** (`/`)
   - プロンプト（質問やタスク）を入力
   - "Generate"ボタンをクリック

2. **自動判定**
   - 信頼度が高い場合 → 自動公開
   - 信頼度が低い場合 → レビュー待ちキューに追加

3. **レビュー画面** (`/review`)
   - レビュー待ちアイテムを確認
   - Approve / Edit / Reject を選択

4. **公開済み画面** (`/published`)
   - 承認された回答を確認

## ファイル構成

```
hitl_llm/
├── hitl_llm.py          # メインアプリケーション
├── hitl_db.json         # データベース（自動生成）
└── README.md            # このファイル
```

## カスタマイズ

### LLMモデルの変更

```python
def llm_generate(prompt: str) -> str:
    """Call local Ollama model."""
    res = ollama_client.generate(
        model="qwen3:4b",  # ← ここを変更
        prompt=prompt
    )
    return res.get("response", "")
```

**推奨モデル:**
- `qwen3:4b`: 軽量・高速
- `qwen3:8b`: バランス型
- `llama3.1:8b`: 英語に強い
- `gemma2:9b`: Google製

### 信頼度閾値の調整

```python
# 3) Threshold -> if low confidence, queue for human review
THRESHOLD = 0.8  # ← ここを変更（0.0〜1.0）
```

- **0.8（デフォルト）**: 厳しい基準（多くがレビュー待ちに）
- **0.5**: 中程度
- **0.3**: 緩い基準（ほとんど自動承認）

### ポート番号の変更

```python
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5001,  # ← ここを変更
        debug=True
    )
```

### 自己評価プロンプトの改善

```python
def llm_self_eval(text: str) -> float:
    """Ask the local LLM for a confidence rating 0.0〜1.0."""
    eval_prompt = f"""あなたは自己評価エンジンです。
以下の回答の信頼度を0.0〜1.0で1つの数値のみ返してください。

評価基準:
- 1.0: 完全に正確で詳細
- 0.5: 部分的に正確
- 0.0: 不正確または不明確

回答:{text}

信頼度（数値のみ）:"""
    res = ollama_client.generate(model="qwen3:4b", prompt=eval_prompt)
    try:
        return float(res.get("response", "0.5").strip())
    except:
        return 0.5
```

## データベース構造

TinyDB（JSON）を使用した軽量データベース：

### pending_table（レビュー待ち）
```json
{
  "id": "abc123",
  "prompt": "質問内容",
  "output": "LLMの回答",
  "confidence": 0.65,
  "created_at": null
}
```

### published_table（公開済み）
```json
{
  "id": "def456",
  "prompt": "質問内容",
  "output": "最終的な回答",
  "confidence": 0.65,
  "human_id": "human_editor"
}
```

## 本番環境への展開

このデモアプリは教育目的です。本番環境では以下を追加してください：

### セキュリティ
- [ ] ユーザー認証（Flask-Login、OAuth）
- [ ] CSRF保護（Flask-WTF）
- [ ] HTTPS/SSL
- [ ] API認証トークン

### データベース
- [ ] PostgreSQL/MySQL へ移行
- [ ] SQLAlchemy による ORM
- [ ] データベースマイグレーション

### 監視・ログ
- [ ] 監査ログ（誰が何を承認/編集/却下したか）
- [ ] エラートラッキング（Sentry）
- [ ] パフォーマンス監視

### スケーラビリティ
- [ ] Celery/Redis によるタスクキュー
- [ ] Webhook 通知（Slack/Email）
- [ ] RBAC（Role-Based Access Control）
- [ ] キャッシング（Redis）

### WSGI サーバー

開発サーバーの代わりに本番用WSGIサーバーを使用：

```bash
# Gunicornをインストール
pip install gunicorn

# 起動
gunicorn -w 4 -b 0.0.0.0:5001 hitl_llm:app
```

## トラブルシューティング

### ポート競合エラー
```
Address already in use
```

**解決策**: ポート番号を変更（5001 → 5002など）

### Ollamaモデルが見つからない
```
Error: model 'qwen3:4b' not found
```

**解決策**:
```bash
ollama pull qwen3:4b
ollama list  # インストール確認
```

### 信頼度が常に0.5になる
**原因**: LLMが数値のみを返していない

**解決策**: 
1. プロンプトを改善
2. 正規表現で数値を抽出
3. より優れたモデルを使用

### データベースが破損
```bash
# データベースファイルを削除して再作成
rm hitl_db.json
python hitl_llm.py
```

## 技術スタック

| 技術 | 用途 | バージョン |
|------|------|------------|
| **Flask** | Webフレームワーク | 3.1.2 |
| **Ollama** | ローカルLLM実行 | 0.6.1 |
| **TinyDB** | 軽量JSONデータベース | 4.8.2 |
| **Python** | プログラミング言語 | 3.9+ |

## アーキテクチャ図

```
┌─────────────┐
│   Browser   │
└──────┬──────┘
       │ HTTP
       ↓
┌─────────────────────────────────────┐
│         Flask Web App               │
│  ┌──────────────────────────────┐  │
│  │  Routes                       │  │
│  │  - /submit  (POST)            │  │
│  │  - /review  (GET)             │  │
│  │  - /approve (POST)            │  │
│  │  - /edit    (GET/POST)        │  │
│  │  - /reject  (POST)            │  │
│  │  - /published (GET)           │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │  LLM Functions                │  │
│  │  - llm_generate()             │  │
│  │  - llm_self_eval()            │  │
│  └──────────────────────────────┘  │
└──────┬─────────────┬────────────────┘
       │             │
       ↓             ↓
┌─────────────┐ ┌──────────────┐
│   Ollama    │ │   TinyDB     │
│  (qwen3:4b) │ │ (hitl_db.json)│
└─────────────┘ └──────────────┘
```

## ユースケース

### 1. カスタマーサポート
- 自動回答が不確実な問い合わせを人間オペレーターに回す
- オペレーターが回答を確認・修正してから送信

### 2. コンテンツ生成
- AIが下書きを作成
- 編集者がレビュー・修正
- 承認後に公開

### 3. データラベリング
- AIが初期ラベルを付与
- 専門家が確認・修正
- 高品質データセットの構築

### 4. 法務・医療
- AIがドラフトを作成
- 専門家が法的/医学的観点でレビュー
- 責任ある判断の保証

## ライセンス

MIT License

## 参考資料

- [Flask公式ドキュメント](https://flask.palletsprojects.com/)
- [Ollama公式サイト](https://ollama.ai/)
- [TinyDB ドキュメント](https://tinydb.readthedocs.io/)
- [Human-in-the-Loop機械学習](https://www.manning.com/books/human-in-the-loop-machine-learning)

---

**作成日**: 2025年11月26日  
**バージョン**: 1.0.0  
**作成者**: SugioNakazawa
