# LangChain チェーンアーキテクチャ サンプル集

LangChainの様々なチェーンパターンを実装したサンプルコレクションです。

## 📋 目次

1. [Simple Chain](#1-simple-chain) - 基本的な単一チェーン
2. [Sequential Chain](#2-sequential-chain) - 複数ステップの順次実行
3. [Router Chain](#3-router-chain) - 条件分岐
4. [Transform Chain](#4-transform-chain) - データ変換
5. [Parallel Chain](#5-parallel-chain) - 並列実行
6. [Streaming Chain](#6-streaming-chain) - ストリーミング出力
7. [Memory Chain](#7-memory-chain) - 会話履歴管理
8. [Custom Chain](#8-custom-chain) - カスタムロジック

## 🚀 セットアップ

### 前提条件

- Python 3.8+
- Ollama がローカルで起動している（http://localhost:11434）
- qwen3:8b モデルがインストール済み

### インストール

```bash
# プロジェクトルートに移動
cd /path/to/langchain-sample

# 仮想環境の作成（初回のみ）
python -m venv venv

# 仮想環境のアクティベート
source venv/bin/activate

# チェーンディレクトリに移動
cd langchain_server/chain

# 依存パッケージのインストール
pip install langchain langchain-openai langchain-community
```

**注意**: 
- 仮想環境をアクティベートすると、プロンプトの先頭に `(venv)` が表示されます
- 作業を終了する際は `deactivate` コマンドで仮想環境を無効化できます
- 次回以降は `source venv/bin/activate` のみでOKです

### Ollama セットアップ

```bash
# Ollama起動
ollama serve

# モデルダウンロード（別ターミナルで）
ollama pull qwen3:8b
```

## 🎯 使い方

### インタラクティブモード

```bash
python chain.py
```

実行すると、以下のメニューが表示されます：

```
実行するサンプルを選択してください:
1. Simple Chain
2. Sequential Chain
3. Router Chain
4. Transform Chain
5. Parallel Chain
6. Streaming Chain
7. Memory Chain
8. Custom Chain
0. 全て実行

番号を入力 (0-8):
```

### プログラムから使用

```python
import asyncio
from chain import simple_chain_example

# 特定のサンプルを実行
asyncio.run(simple_chain_example())
```

## 📚 各チェーンの説明

### 1. Simple Chain

**用途**: 最も基本的な単一タスクの実行

**特徴**:
- プロンプト → LLM → 出力パーサー の単純な流れ
- LCEL (LangChain Expression Language) を使用
- パイプライン演算子 `|` で連結

**実例**: 翻訳タスク

```python
chain = prompt | llm | StrOutputParser()
result = await chain.ainvoke({"target_lang": "英語", "text": "こんにちは"})
```

### 2. Sequential Chain

**用途**: 複数のステップを順番に実行

**特徴**:
- 前のステップの出力が次のステップの入力になる
- ブログ記事生成などの段階的な作業に適している
- 各ステップの結果を保持

**実例**: ブログ記事作成
1. キーワード → タイトル生成
2. タイトル → 見出し作成
3. タイトル + 見出し → 本文作成

### 3. Router Chain

**用途**: 入力内容に応じて処理を分岐

**特徴**:
- 分類器で入力カテゴリを判定
- カテゴリごとに異なる処理ルートを実行
- 動的なワークフロー制御

**実例**: 質問応答システム
- 技術的質問 → 技術的な回答
- ビジネス質問 → ビジネス視点の回答
- 日常的質問 → カジュアルな回答

### 4. Transform Chain

**用途**: データの前処理・後処理を含むチェーン

**特徴**:
- カスタム変換関数を組み込み
- データクリーニング、正規化
- メタデータの追加・抽出

**実例**: テキスト要約
1. テキストのクリーニング
2. 単語数カウント
3. 要約生成
4. メタデータ付加

### 5. Parallel Chain

**用途**: 複数のタスクを同時並行で実行

**特徴**:
- 独立したタスクを並列実行
- 実行時間の短縮
- `RunnableParallel` を使用

**実例**: テキスト分析
- 感情分析
- カテゴリ分類
- キーワード抽出
- 要約生成

すべて同時に実行し、結果を統合

### 6. Streaming Chain

**用途**: リアルタイムで結果を受信

**特徴**:
- 生成中のテキストを逐次表示
- ユーザー体験の向上
- チャットボットに最適

**実例**: 長文生成
```python
async for chunk in chain.astream(input):
    print(chunk, end='', flush=True)
```

### 7. Memory Chain

**用途**: 会話履歴を保持

**特徴**:
- 過去の対話内容を記憶
- コンテキストを考慮した応答
- `ConversationBufferMemory` を使用

**実例**: 対話システム
- ユーザー名や好みを記憶
- 前の会話を参照
- 文脈に沿った応答

### 8. Custom Chain

**用途**: 複雑なビジネスロジックの実装

**特徴**:
- カスタム検証ロジック
- 条件付き実行
- エラーハンドリング

**実例**: クエリ処理システム
1. 入力検証
2. クエリ拡張
3. 条件付き実行
4. エラーレポート

## 🎨 LCEL (LangChain Expression Language)

### パイプライン演算子

```python
# 基本的な連結
chain = prompt | llm | parser

# 複雑な構造
chain = (
    input_processor
    | RunnableParallel(
        task1=chain1,
        task2=chain2
    )
    | output_formatter
)
```

### Runnable コンポーネント

| コンポーネント | 用途 |
|---------------|------|
| `RunnablePassthrough` | 入力をそのまま通過 |
| `RunnableLambda` | カスタム関数を実行 |
| `RunnableParallel` | 並列実行 |
| `RunnableBranch` | 条件分岐 |

## 📊 パフォーマンス比較

| チェーンタイプ | 実行時間 | 用途 | 複雑度 |
|---------------|---------|------|--------|
| Simple | 最速 | 単一タスク | ⭐ |
| Sequential | 中速 | 段階的処理 | ⭐⭐ |
| Parallel | 高速 | 独立タスク | ⭐⭐⭐ |
| Router | 中速 | 動的分岐 | ⭐⭐⭐ |
| Custom | 可変 | 複雑なロジック | ⭐⭐⭐⭐ |

## 🛠️ カスタマイズ

### LLMモデルの変更

```python
def get_llm(streaming: bool = False, temperature: float = 0.7):
    return ChatOpenAI(
        openai_api_base="http://localhost:11434/v1",
        model="llama3:latest",  # モデルを変更
        temperature=temperature,
        openai_api_key="EMPTY"
    )
```

### プロンプトのカスタマイズ

```python
custom_prompt = ChatPromptTemplate.from_template(
    "あなたは{role}です。{task}"
)
```

### カスタムパーサーの追加

```python
from langchain_core.output_parsers import BaseOutputParser

class CustomParser(BaseOutputParser):
    def parse(self, text: str):
        # カスタムパース処理
        return {"result": text.strip()}
```

## 🐛 トラブルシューティング

### Ollama接続エラー

```bash
# Ollamaが起動しているか確認
curl http://localhost:11434/api/tags

# Ollama再起動
ollama serve
```

### モデルが見つからない

```bash
# 利用可能なモデルを確認
ollama list

# モデルをダウンロード
ollama pull qwen3:8b
```

### メモリ不足エラー

```python
# より小さいモデルを使用
model="qwen3:4b"

# または temperature を調整
temperature=0.3
```

### ストリーミングが動作しない

```python
# streaming=True を明示的に設定
llm = ChatOpenAI(streaming=True, ...)

# astream() メソッドを使用
async for chunk in chain.astream(input):
    print(chunk)
```

## 📖 応用例

### 1. カスタマーサポートボット

```python
# Router + Memory Chain の組み合わせ
support_chain = router_chain | memory_chain
```

### 2. コンテンツ生成パイプライン

```python
# Sequential + Parallel の組み合わせ
content_pipeline = sequential_chain | parallel_chain
```

### 3. データ分析レポート

```python
# Transform + Parallel の組み合わせ
analysis_chain = transform_chain | parallel_analysis
```

## 🔗 関連リソース

- [LangChain公式ドキュメント](https://python.langchain.com/)
- [LCEL ガイド](https://python.langchain.com/docs/expression_language/)
- [Ollama ドキュメント](https://ollama.ai/docs)

## 📝 ベストプラクティス

1. **シンプルから始める**: まずは Simple Chain で基本を理解
2. **適切なチェーンを選ぶ**: タスクに応じて最適なパターンを使用
3. **エラーハンドリング**: try-catch でエラーを適切に処理
4. **ストリーミング活用**: ユーザー体験向上のため積極的に使用
5. **メモリ管理**: 長い会話では適切なメモリ管理を実装

## 🤝 貢献

改善案やバグ報告は Issue または Pull Request でお願いします。

## 📄 ライセンス

MIT License

---

**最終更新**: 2025年11月10日  
**バージョン**: 1.0.0  
**作成者**: SugioNakazawa
