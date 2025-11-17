"""
LangChain RAG with PDF - PDFドキュメントを使用したRAGシステム

このスクリプトは、PDFファイルを読み込んでRAGシステムを構築します：
1. PDFファイルの読み込み
2. テキスト抽出とチャンク分割
3. ベクトルストアの構築
4. 質問応答システム
"""

import asyncio
import os
from pathlib import Path
from typing import List
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===== 設定 =====
def get_embeddings():
    """Embedding モデルを取得"""
    return OllamaEmbeddings(
        base_url="http://localhost:11434",
        model="kun432/cl-nagoya-ruri-large"
    )
        # model="kun432/cl-nagoya-ruri-large"
        # model="mxbai-embed-large"

def get_llm(streaming: bool = False):
    """LLMモデルを取得"""
    return ChatOpenAI(
        openai_api_base="http://localhost:11434/v1",
        streaming=streaming,
        temperature=0.3,
        openai_api_key="EMPTY",
        model="qwen3:8b"
        # max_tokens=2000
    )
        # model="qwen3:30b-a3b"
        # model="gpt-oss:20b"
        # model="qwen3:14b"
        # model="qwen3:8b"
        # model="qwen3:4b"

# ===== PDFディレクトリの設定 =====
PDF_DIR = Path(__file__).parent / "documents"

def ensure_pdf_directory():
    """PDFディレクトリが存在することを確認"""
    PDF_DIR.mkdir(exist_ok=True)
    return PDF_DIR

# ===== サンプルPDFの作成 =====
def create_sample_pdf():
    """サンプルPDFファイルを作成（PDFがない場合）- 日本語対応版"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        
        sample_pdf_path = PDF_DIR / "sample_tech_article.pdf"
        
        if sample_pdf_path.exists():
            return sample_pdf_path
        
        print(f"📝 サンプルPDFを作成中: {sample_pdf_path}")
        
        # 日本語フォントを登録
        pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3'))
        
        c = canvas.Canvas(str(sample_pdf_path), pagesize=letter)
        width, height = letter
        
        # タイトル
        c.setFont("HeiseiMin-W3", 20)
        c.drawString(100, height - 100, "人工知能と機械学習の概要")
        
        # 本文
        c.setFont("HeiseiMin-W3", 12)
        y_position = height - 150
        
        content = [
            "人工知能入門",
            "",
            "人工知能（AI）は、人間のように考え学習できる知的な機械を作成することを",
            "目的としたコンピュータサイエンスの分野です。機械学習（ML）は、経験を通じて",
            "改善するアルゴリズムに焦点を当てたAIのサブセットです。",
            "",
            "機械学習の種類：",
            "1. 教師あり学習 - ラベル付きデータから学習",
            "2. 教師なし学習 - ラベルなしデータのパターンを発見",
            "3. 強化学習 - 試行錯誤を通じて学習",
            "",
            "ディープラーニング：",
            "ディープラーニングは、複数の層を持つニューラルネットワークを使用して",
            "複雑なデータを処理します。人気のあるアーキテクチャには、画像処理用のCNN、",
            "系列データ用のRNN、自然言語処理用のTransformerなどがあります。",
            "",
            "応用分野：",
            "- 画像認識とコンピュータビジョン",
            "- 自然言語処理",
            "- 音声認識",
            "- 自動運転車",
            "- レコメンデーションシステム",
            "",
            "PythonとRAG：",
            "Pythonは、AI・ML開発で最も人気のあるプログラミング言語です。",
            "LangChainを使用すると、大規模言語モデルと外部知識を組み合わせた",
            "RAG（Retrieval-Augmented Generation）システムを構築できます。",
            "RAGは、ベクトルデータベースから関連情報を検索し、それをもとに",
            "精度の高い回答を生成する技術です。",
        ]
        
        for line in content:
            c.drawString(100, y_position, line)
            y_position -= 20
            if y_position < 100:
                c.showPage()
                c.setFont("HeiseiMin-W3", 12)
                y_position = height - 100
        
        c.save()
        print(f"✅ サンプルPDF作成完了: {sample_pdf_path}")
        return sample_pdf_path
        
    except ImportError:
        print("⚠️  reportlabがインストールされていません。")
        print("   pip install reportlab でインストールしてください。")
        return None

# ===== 1. PDFファイルの読み込み =====
async def load_pdf_documents():
    """PDFファイルを読み込む"""
    print("\n" + "="*70)
    print("1️⃣  PDFファイルの読み込み")
    print("="*70)
    
    pdf_dir = ensure_pdf_directory()
    
    # PDFファイルの確認
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  {pdf_dir} にPDFファイルがありません。")
        print("サンプルPDFを作成します...")
        sample_pdf = create_sample_pdf()
        if sample_pdf:
            pdf_files = [sample_pdf]
        else:
            print("\n手動でPDFファイルを配置してください:")
            print(f"  {pdf_dir}/your_document.pdf")
            return None
    
    print(f"📚 PDFディレクトリ: {pdf_dir}")
    print(f"📄 見つかったPDFファイル: {len(pdf_files)}件")
    
    documents = []
    
    for pdf_file in pdf_files:
        print(f"\n📖 読み込み中: {pdf_file.name}")
        
        try:
            # PDFローダーの作成
            loader = PyPDFLoader(str(pdf_file))
            
            # ドキュメントの読み込み
            docs = loader.load()
            
            print(f"   ページ数: {len(docs)}")
            
            # メタデータの追加
            for i, doc in enumerate(docs):
                doc.metadata["source"] = pdf_file.name
                doc.metadata["page"] = i + 1
            
            documents.extend(docs)
            
            # 最初のページのプレビュー
            if docs:
                preview = docs[0].page_content[:200].replace('\n', ' ')
                print(f"   プレビュー: {preview}...")
                
        except Exception as e:
            print(f"   ❌ エラー: {e}")
            continue
    
    print(f"\n✅ 合計 {len(documents)} ページを読み込みました")
    return documents

# ===== 2. ディレクトリ内の全PDFを読み込み =====
async def load_pdf_directory():
    """ディレクトリ内の全PDFを一括読み込み"""
    print("\n" + "="*70)
    print("2️⃣  ディレクトリローダー（一括読み込み）")
    print("="*70)
    
    pdf_dir = ensure_pdf_directory()
    
    # ディレクトリローダーの作成
    loader = DirectoryLoader(
        str(pdf_dir),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    
    print(f"📂 読み込み中: {pdf_dir}")
    
    try:
        documents = loader.load()
        print(f"✅ {len(documents)} ページを読み込みました")
        return documents
    except Exception as e:
        print(f"❌ エラー: {e}")
        return []

# ===== 3. PDFからベクトルストアを構築 =====
async def build_vectorstore_from_pdf(documents):
    """PDFドキュメントからベクトルストアを構築"""
    print("\n" + "="*70)
    print("3️⃣  ベクトルストアの構築")
    print("="*70)
    
    if not documents:
        print("❌ ドキュメントがありません")
        return None
    
    # テキスト分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", ". ", " "]
    )
    
    # ドキュメントを分割
    print(f"📄 元のページ数: {len(documents)}")
    splits = text_splitter.split_documents(documents)
    print(f"📝 分割後のチャンク数: {len(splits)}")
    
    # サンプルチャンクの表示
    if splits:
        print(f"\n🔍 サンプルチャンク:")
        sample = splits[0]
        print(f"   内容: {sample.page_content[:150]}...")
        print(f"   メタデータ: {sample.metadata}")
    
    # Embeddingモデル
    embeddings = get_embeddings()
    
    # ベクトルストアの作成
    print("\n🔄 ベクトル化中...")
    vectorstore = await FAISS.afrom_documents(splits, embeddings)
    print("✅ ベクトルストア構築完了")
    
    return vectorstore

# ===== 4. PDFベースのRAG質問応答 =====
async def pdf_rag_qa(vectorstore):
    """PDFベースのRAG質問応答システム"""
    print("\n" + "="*70)
    print("4️⃣  PDF RAG 質問応答システム")
    print("="*70)
    
    if not vectorstore:
        print("❌ ベクトルストアがありません")
        return
    
    # Retriever の作成
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # プロンプトテンプレート
    template = """以下のPDFドキュメントの内容を参考に、質問に答えてください。
回答には必ず参照したページ番号を含めてください。

参考情報:
{context}

質問: {question}

回答:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # RAGチェーンの構築
    def format_docs(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', 'unknown')
            formatted.append(f"[{source} - ページ{page}]\n{doc.page_content}")
        return "\n\n".join(formatted)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    
    # 質問リスト
    questions = [
        "人工知能とは何ですか？",
        "機械学習の種類を教えてください",
        "RAGとは何ですか？",
        "ディープラーニングの特徴は何ですか？"
    ]
    
    for question in questions:
        print(f"\n❓ 質問: {question}")
        
        # 検索されるドキュメントを表示
        retrieved_docs = await retriever.ainvoke(question)
        print(f"📚 参照ドキュメント:")
        for doc in retrieved_docs:
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', 'unknown')
            print(f"   - {source} (ページ {page})")
        
        # 回答生成
        answer = await rag_chain.ainvoke(question)
        print(f"💡 回答:\n{answer}")
        print("-" * 70)

# ===== 5. ストリーミングPDF RAG =====
async def streaming_pdf_rag(vectorstore):
    """ストリーミング出力でのPDF RAG"""
    print("\n" + "="*70)
    print("5️⃣  ストリーミングPDF RAG")
    print("="*70)
    
    if not vectorstore:
        print("❌ ベクトルストアがありません")
        return
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    template = """以下のPDFドキュメントの内容を参考に、質問に詳しく答えてください。

参考情報:
{context}

質問: {question}

回答:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', 'unknown')
            formatted.append(f"[{source} p.{page}]\n{doc.page_content}")
        return "\n\n".join(formatted)
    
    streaming_llm = get_llm(streaming=True)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | streaming_llm
        | StrOutputParser()
    )
    
    question = "ディープラーニングとそのアーキテクチャについて詳しく説明してください。"
    print(f"❓ 質問: {question}\n")
    print("💡 回答（ストリーミング）:")
    print("   ", end="", flush=True)
    
    async for chunk in rag_chain.astream(question):
        print(chunk, end="", flush=True)
    
    print("\n\n✅ ストリーミング完了")

# ===== 6. インタラクティブな質問応答 =====
async def interactive_pdf_qa(vectorstore):
    """インタラクティブな質問応答モード"""
    print("\n" + "="*70)
    print("6️⃣  インタラクティブ質問応答")
    print("="*70)
    
    if not vectorstore:
        print("❌ ベクトルストアがありません")
        return
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    template = """以下のPDFドキュメントの内容を参考に、質問に答えてください。

参考情報:
{context}

質問: {question}

回答:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', 'unknown')
            formatted.append(f"[{source} p.{page}]\n{doc.page_content}")
        return "\n\n".join(formatted)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    
    print("\n💬 PDFに関する質問を入力してください（'quit'で終了）")
    print("-" * 70)
    
    while True:
        try:
            question = input("\n❓ 質問: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 終了します")
                break
            
            if not question:
                continue
            
            print("\n🔍 検索中...")
            retrieved_docs = await retriever.ainvoke(question)
            
            print("📚 参照ドキュメント:")
            for doc in retrieved_docs:
                source = doc.metadata.get('source', 'unknown')
                page = doc.metadata.get('page', 'unknown')
                preview = doc.page_content[:100].replace('\n', ' ')
                print(f"   - {source} (p.{page}): {preview}...")
            
            print("\n💡3 回答:")
            answer = await rag_chain.ainvoke(question)
            print(answer)
            print("-" * 70)
            
        except KeyboardInterrupt:
            print("\n👋 終了します")
            break
        except Exception as e:
            print(f"❌ エラー: {e}")

# ===== メイン実行 =====
async def main():
    """全てのサンプルを実行"""
    print("\n" + "🌟"*35)
    print("LangChain RAG with PDF")
    print("🌟"*35)
    
    # PDFドキュメントの読み込み
    documents = await load_pdf_documents()
    
    if not documents:
        print("\n❌ PDFファイルを読み込めませんでした")
        return
    
    # ベクトルストアの構築
    vectorstore = await build_vectorstore_from_pdf(documents)
    
    if not vectorstore:
        print("\n❌ ベクトルストアを構築できませんでした")
        return
    
    examples = [
        ("PDF RAG 質問応答", lambda: pdf_rag_qa(vectorstore)),
        ("ストリーミングPDF RAG", lambda: streaming_pdf_rag(vectorstore)),
        ("インタラクティブ質問応答", lambda: interactive_pdf_qa(vectorstore)),
    ]
    
    print("\n実行するサンプルを選択してください:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    print("0. 全て実行（インタラクティブ除く）")
    
    try:
        choice = input("\n番号を入力 (0-3): ").strip()
        
        if choice == "0":
            for i, (name, func) in enumerate(examples):
                if i < len(examples) - 1:  # インタラクティブ除く
                    await func()
        elif choice.isdigit() and 1 <= int(choice) <= len(examples):
            _, func = examples[int(choice) - 1]
            await func()
        else:
            print("❌ 無効な選択です")
            return
            
    except KeyboardInterrupt:
        print("\n\n⚠️  実行が中断されました")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "🎉"*35)
    print("サンプル実行完了！")
    print("🎉"*35 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
