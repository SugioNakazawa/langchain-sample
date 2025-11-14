# Appendix by Human
LangChain での RAG 利用。

1. 読み込ませたい PDFドキュメントを langchain_server/rag/documents に配置。空の時には作成します。
2. Python仮想環境で rag_with_pdf.py を実行し、「3. インタラクティブ質問応答」を選択して質問を入力。中止の時は「quit」を入力。

```bash
$ cd langchain-sample
$ source venv/bin/activate
(venv) $ python langchain_server/rag/rag_with_pdf.py
```

# LangChain RAG (Retrieval-Augmented Generation) サンプル by Copilot

LangChainのembedding機能を活用したRAGシステムの実装サンプル集です。

## 📋 目次

1. [embedding_basic.py](#embedding_basicpy) - Embedding基本操作
2. [rag_complete.py](#rag_completepy) - 完全なRAGシステム
3. [rag_with_pdf.py](#rag_with_pdfpy) - PDFドキュメントを使用したRAG
4. [rag_advanced.py](#rag_advancedpy) - 高度なRAG技術（予定）

## 🚀 セットアップ

### 前提条件

- Python 3.8+
- Ollama がローカルで起動している（http://localhost:11434）
- 必要なモデル:
  - `mxbai-embed-large` - Embeddingモデル
  - `mxbai-embed-large` - Embeddingモデル
  - `qwen3:8b` - LLMモデル
  - `qwen3:8b` - LLMモデル

### Ollamaモデルのインストール

```bash
# Embeddingモデル
ollama pull mxbai-embed-large

# LLMモデル（既にある場合はスキップ）
ollama pull qwen3:8b
```

### 仮想環境のセットアップ

```bash
# プロジェクトルートに移動
cd /path/to/langchain-sample

# 仮想環境の作成（初回のみ）
python -m venv venv

# 仮想環境のアクティベート
source venv/bin/activate

# RAGディレクトリに移動
cd langchain_server/rag

# 必要なパッケージのインストール
pip install langchain langchain-openai langchain-community faiss-cpu numpy pypdf reportlab
```

**注意**: 
- 仮想環境は `venv` ディレクトリを使用します
- プロンプトに `(venv)` が表示されたらアクティベート成功です
- 作業終了時は `deactivate` で仮想環境を無効化できます
- 次回以降は `source venv/bin/activate` のみでOK
- GPU版を使用する場合は `faiss-cpu` を `faiss-gpu` に変更してください

## 📚 サンプルファイルの説明

### embedding_basic.py

**概要**: Embeddingの基本的な使い方を学ぶためのサンプル

**内容**:
1. ✅ 基本的なテキストのベクトル化
2. ✅ 複数テキストのバッチ処理
3. ✅ コサイン類似度計算
4. ✅ 類似度検索
5. ✅ セマンティック検索
6. ✅ 多言語embedding

**実行方法**:
```bash
python embedding_basic.py
```

**学べること**:
- テキストをベクトルに変換する方法
- ベクトル間の類似度計算
- 意味的に近いテキストの検索
- 多言語テキストの扱い

**出力例**:
```
🔍 クエリ: ペットについて教えて

🎯 類似度ランキング（上位5件）:
  1位 (類似度: 0.8521): 猫は可愛いペットです。
  2位 (類似度: 0.8314): 犬は忠実な友達です。
  3位 (類似度: 0.5621): 鳥は空を飛びます。
```

### rag_complete.py

**概要**: 完全なRAGシステムの実装例

**内容**:
1. ✅ ドキュメントの準備とベクトルストア構築
2. ✅ 類似ドキュメント検索
3. ✅ スコア付き検索
4. ✅ RAGチェーンによる質問応答
5. ✅ ストリーミングRAG
6. ✅ MMR検索（多様性考慮）

**実行方法**:
```bash
python rag_complete.py
```

**学べること**:
- ドキュメントのチャンク分割
- FAISSベクトルストアの使用
- Retrieverの設定
- LLMと組み合わせた質問応答
- ストリーミング出力
- 多様性を考慮した検索（MMR）

**出力例**:
```
❓ 質問: Pythonはいつ開発されましたか？
📚 参照ドキュメント数: 3
💡 回答: Pythonは1991年にGuido van Rossumによって開発されました。
        読みやすく、書きやすい構文が特徴で、初心者にも扱いやすい
        言語として広く使われています。
```

### rag_with_pdf.py

**概要**: PDFファイルを読み込んで使用するRAGシステム

**内容**:
1. ✅ PDFファイルの読み込み
2. ✅ ディレクトリ内の全PDF一括読み込み
3. ✅ PDFからベクトルストア構築
4. ✅ PDFベースの質問応答
5. ✅ ストリーミングPDF RAG
6. ✅ インタラクティブ質問応答

**実行方法**:
```bash
python rag_with_pdf.py
```

**PDFファイルの配置**:
- 自動的に `documents/` ディレクトリが作成されます
- PDFファイルがない場合、サンプルPDFが自動生成されます
- 独自のPDFファイルを `documents/` に配置できます

**学べること**:
- PyPDFLoaderを使ったPDF読み込み
- DirectoryLoaderによる一括読み込み
- PDFメタデータの活用（ページ番号など）
- ページ情報を含む回答生成
- インタラクティブな質問応答システム

**出力例**:
```
❓ 質問: What is Artificial Intelligence?
📚 参照ドキュメント:
   - sample_tech_article.pdf (ページ 1)
💡 回答: 人工知能（AI）は、コンピュータ科学の一分野であり、
        人間のように思考・学習ができる知能を持つ機械を創造する
        ことを目指す分野です。
        [参照: sample_tech_article.pdf - ページ1]
```

**内容**:
1. ✅ ドキュメントの準備とベクトルストア構築
2. ✅ 類似ドキュメント検索
3. ✅ スコア付き検索
4. ✅ RAGチェーンによる質問応答
5. ✅ ストリーミングRAG
6. ✅ MMR検索（多様性考慮）

**実行方法**:
```bash
python rag_complete.py
```

**学べること**:
- ドキュメントのチャンク分割
- FAISSベクトルストアの使用
- Retrieverの設定
- LLMと組み合わせた質問応答
- ストリーミング出力
- 多様性を考慮した検索（MMR）

**出力例**:
```
❓ 質問: Pythonはいつ開発されましたか？
📚 参照ドキュメント数: 3
💡 回答: Pythonは1991年にGuido van Rossumによって開発されました。
        読みやすく、書きやすい構文が特徴で、初心者にも扱いやすい
        言語として広く使われています。
```

## 🎯 主な機能

### 1. Embedding（ベクトル化）

テキストを高次元ベクトルに変換し、意味的な類似性を数値化：

```python
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector = await embeddings.aembed_query("こんにちは")
```

### 2. ベクトルストア

ベクトルを効率的に保存・検索：

```python
vectorstore = await FAISS.afrom_documents(documents, embeddings)
results = await vectorstore.asimilarity_search("質問", k=3)
```

### 3. RAGチェーン

検索と生成を組み合わせた質問応答：

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

## 🛠️ カスタマイズ

### Embeddingモデルの変更

```python
# 別のモデルを使用
embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="all-minilm"  # より軽量なモデル
)
```

### チャンクサイズの調整

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # チャンクサイズを増やす
    chunk_overlap=100,   # オーバーラップを増やす
    separators=["\n\n", "\n", "。", " "]
)
```

### 検索結果数の変更

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # 上位5件を取得
)
```

### 検索タイプの変更

```python
# MMR検索（多様性重視）
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20,
        "lambda_mult": 0.5  # 0=多様性、1=類似性
    }
)

# スコア閾値による検索
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.8,
        "k": 5
    }
)
```

## 📊 パフォーマンス

### ベクトルストアの比較

| ベクトルストア | 速度 | メモリ | 永続化 | 用途 |
|---------------|------|--------|--------|------|
| **FAISS** | 高速 | 中 | ファイル | ローカル開発・プロトタイプ |
| **Chroma** | 中速 | 低 | DB | 小〜中規模アプリ |
| **Pinecone** | 高速 | クラウド | クラウド | 本番環境・大規模 |
| **Weaviate** | 高速 | 中 | DB | 本番環境・中〜大規模 |

### 推奨設定

**小規模（〜1000ドキュメント）**:
```python
chunk_size = 500
chunk_overlap = 50
retriever_k = 3
```

**中規模（〜10000ドキュメント）**:
```python
chunk_size = 1000
chunk_overlap = 100
retriever_k = 5
use_mmr = True
```

**大規模（10000+ドキュメント）**:
```python
# FAISSの代わりにChromaやPineconeを検討
chunk_size = 1000
chunk_overlap = 200
retriever_k = 10
use_hybrid_search = True  # キーワード+ベクトル
```

## 🐛 トラブルシューティング

### Ollamaモデルが見つからない

```bash
# モデル一覧を確認
ollama list

# モデルをダウンロード
ollama pull mxbai-embed-large
```

### FAISSインストールエラー

```bash
# CPU版
pip install faiss-cpu

# GPU版（CUDAが必要）
pip install faiss-gpu
```

### メモリ不足エラー

```python
# チャンクサイズを大きくしてチャンク数を減らす
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # より大きく
    chunk_overlap=50   # オーバーラップを減らす
)

# またはバッチサイズを調整
vectorstore = await FAISS.afrom_documents(
    documents[:100],  # 最初の100件のみ処理
    embeddings
)
```

### 検索精度が低い

1. **チャンクサイズの調整**: 小さすぎると文脈が失われる
2. **オーバーラップの増加**: 重要な情報が分割されないように
3. **retriever_kの増加**: より多くのドキュメントを参照
4. **MMR検索の使用**: 多様な情報を取得
5. **Embeddingモデルの変更**: より高性能なモデルを試す

## 💡 ベストプラクティス

### 1. 適切なチャンク分割

```python
# ドキュメントの種類に応じて区切り文字を調整
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=[
        "\n\n",  # 段落
        "\n",    # 行
        "。",    # 日本語の文
        "、",    # 日本語の句読点
        " ",     # スペース
        ""       # 文字
    ]
)
```

### 2. メタデータの活用

```python
documents = [
    Document(
        page_content=text,
        metadata={
            "source": "article.pdf",
            "page": 1,
            "author": "John Doe",
            "date": "2024-01-01"
        }
    )
]

# メタデータでフィルタリング
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"author": "John Doe"}
    }
)
```

### 3. プロンプトの最適化

```python
template = """以下のコンテキストを参考に質問に答えてください。
コンテキストに答えがない場合は、「提供された情報からは答えられません」と答えてください。
コンテキストから推測される情報も含めて、具体的かつ詳細に説明してください。

コンテキスト:
{context}

質問: {question}

回答（日本語で詳しく）:"""
```

### 4. エラーハンドリング

```python
try:
    results = await vectorstore.asimilarity_search(query, k=3)
    if not results:
        print("関連するドキュメントが見つかりませんでした")
except Exception as e:
    print(f"検索エラー: {e}")
    # フォールバック処理
```

## 🔗 関連リソース

- [LangChain公式ドキュメント](https://python.langchain.com/)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [Ollama公式サイト](https://ollama.ai/)
- [RAG解説記事](https://python.langchain.com/docs/use_cases/question_answering/)

## 🌏 日本語ドキュメント対応

### PDF日本語フォント対応

`rag_with_pdf.py`は日本語PDFに完全対応しています：

```python
# 日本語フォントを使用したPDF生成
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3'))
```

**自動生成されるサンプルPDF**:
- 日本語で書かれたAI・機械学習の解説
- 日本語質問に対応した質問応答
- ページ参照機能付き回答生成

**使用例**:
```bash
# 日本語対応RAGの実行
cd langchain_server/rag
source ../../venv/bin/activate
python rag_with_pdf.py
```

### 日本語Embedding

日本語に最適化されたEmbeddingモデル：

```python
# 日本語対応モデル
embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="kun432/cl-nagoya-ruri-large"  # 日本語特化モデル
)
```

**推奨モデル**:
- `kun432/cl-nagoya-ruri-large`: 日本語特化、高精度
- `mxbai-embed-large`: 多言語対応、汎用性高い

### 日本語質問例

```python
questions = [
    "人工知能とは何ですか？",
    "機械学習の種類を教えてください",
    "RAGとは何ですか？",
    "ディープラーニングの特徴は何ですか？"
]
```

## 📝 次のステップ

1. **ドキュメントローダー**: PDFやWebページからのドキュメント読み込み ✅
2. **Hybrid Search**: キーワード検索とベクトル検索の組み合わせ
3. **Re-ranking**: 検索結果の再ランキング
4. **Multi-Query**: 複数の質問バリエーションで検索
5. **Parent Document Retriever**: 大きなコンテキストの取得
6. **日本語ドキュメント対応**: 日本語PDF処理と質問応答 ✅

## 🤝 貢献

改善案やバグ報告は Issue または Pull Request でお願いします。

---

**最終更新**: 2025年11月14日  
**バージョン**: 1.0.0  
**作成者**: SugioNakazawa
