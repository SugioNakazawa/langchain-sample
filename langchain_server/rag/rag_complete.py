"""
LangChain RAG (Retrieval-Augmented Generation) 完全サンプル

このスクリプトは、embeddingを使用したRAGシステムの実装例です：
1. ドキュメントの読み込みとチャンク分割
2. ベクトルストアの構築
3. 類似ドキュメント検索
4. LLMと組み合わせた質問応答
"""

import asyncio
from typing import List
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
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
        model="mxbai-embed-large"
    )

def get_llm():
    """LLMモデルを取得"""
    return ChatOpenAI(
        openai_api_base="http://localhost:11434/v1",
        temperature=0.7,
        openai_api_key="EMPTY",
        model="qwen3:8b"
    )

# ===== サンプルドキュメント =====
SAMPLE_DOCUMENTS = [
    """
    Python は、1991年にGuido van Rossumによって開発されたプログラミング言語です。
    読みやすく、書きやすい構文が特徴で、初心者にも扱いやすい言語として広く使われています。
    Web開発、データ分析、機械学習、自動化など、様々な分野で活用されています。
    """,
    """
    機械学習は人工知能の一分野で、コンピュータがデータから学習する技術です。
    教師あり学習、教師なし学習、強化学習の3つの主要なアプローチがあります。
    画像認識、音声認識、自然言語処理などの応用があります。
    """,
    """
    深層学習はニューラルネットワークを使用した機械学習の手法です。
    多層のニューラルネットワークにより、複雑なパターンを学習できます。
    CNNは画像処理に、RNNは時系列データに、Transformerは自然言語処理に使われます。
    """,
    """
    自然言語処理（NLP）は、人間の言語をコンピュータで処理する技術です。
    形態素解析、構文解析、意味解析などの基本技術があります。
    機械翻訳、文章要約、感情分析、質問応答システムなどに応用されています。
    """,
    """
    LangChainは、大規模言語モデル（LLM）を使ったアプリケーション開発を
    簡単にするためのフレームワークです。プロンプト管理、チェーン構築、
    メモリ管理、エージェント機能などを提供します。
    """,
    """
    RAG（Retrieval-Augmented Generation）は、検索と生成を組み合わせた技術です。
    外部ナレッジベースから関連情報を検索し、それを元にLLMが回答を生成します。
    これにより、LLMの知識を最新情報で補完できます。
    """,
    """
    ベクトルデータベースは、高次元ベクトルの効率的な保存と検索を行うデータベースです。
    embeddingベクトルを保存し、類似度検索を高速に実行できます。
    FAISS、Chroma、Pinecone、Weaviateなどが代表的な実装です。
    """,
    """
    Transformerアーキテクチャは、2017年に発表された革新的なニューラルネットワークです。
    Self-Attentionメカニズムにより、長距離依存関係を効果的に捉えることができます。
    BERT、GPT、T5など、多くの最新モデルのベースとなっています。
    """
]

# ===== 1. ドキュメント準備とベクトルストア構築 =====
async def build_vector_store():
    """ベクトルストアを構築"""
    print("\n" + "="*70)
    print("1️⃣  ベクトルストアの構築")
    print("="*70)
    
    # ドキュメントオブジェクトの作成
    documents = [
        Document(page_content=text.strip(), metadata={"source": f"doc_{i}"})
        for i, text in enumerate(SAMPLE_DOCUMENTS, 1)
    ]
    
    print(f"📚 ドキュメント数: {len(documents)}")
    
    # テキスト分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    
    # ドキュメントを分割
    splits = text_splitter.split_documents(documents)
    print(f"📄 分割後のチャンク数: {len(splits)}")
    
    # Embeddingモデル
    embeddings = get_embeddings()
    
    # ベクトルストアの作成
    print("🔄 ベクトル化中...")
    vectorstore = await FAISS.afrom_documents(splits, embeddings)
    print("✅ ベクトルストア構築完了")
    
    return vectorstore

# ===== 2. 類似ドキュメント検索 =====
async def similarity_search_demo(vectorstore):
    """類似ドキュメント検索のデモ"""
    print("\n" + "="*70)
    print("2️⃣  類似ドキュメント検索")
    print("="*70)
    
    queries = [
        "Pythonについて教えて",
        "機械学習の種類は？",
        "RAGとは何ですか？"
    ]
    
    for query in queries:
        print(f"\n🔍 クエリ: {query}")
        
        # 類似ドキュメント検索（上位3件）
        docs = await vectorstore.asimilarity_search(query, k=3)
        
        print("📖 検索結果:")
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.replace('\n', ' ')[:100]
            print(f"  {i}. {content}...")
            print(f"     (ソース: {doc.metadata.get('source', 'unknown')})")

# ===== 3. スコア付き検索 =====
async def similarity_search_with_score_demo(vectorstore):
    """スコア付き類似ドキュメント検索"""
    print("\n" + "="*70)
    print("3️⃣  スコア付き検索")
    print("="*70)
    
    query = "ニューラルネットワークの応用"
    print(f"🔍 クエリ: {query}\n")
    
    # スコア付き検索
    docs_with_scores = await vectorstore.asimilarity_search_with_score(query, k=5)
    
    print("📊 検索結果（類似度スコア付き）:")
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        content = doc.page_content.replace('\n', ' ')[:80]
        print(f"  {i}. スコア: {score:.4f}")
        print(f"     内容: {content}...")
        print()

# ===== 4. RAGチェーン：質問応答システム =====
async def rag_chain_demo(vectorstore):
    """RAGを使った質問応答システム"""
    print("\n" + "="*70)
    print("4️⃣  RAGチェーン：質問応答システム")
    print("="*70)
    
    # Retriever の作成
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # プロンプトテンプレート
    template = """以下のコンテキストを使用して質問に答えてください。
コンテキストに情報がない場合は、「わかりません」と答えてください。

コンテキスト:
{context}

質問: {question}

回答:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # RAGチェーンの構築
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    
    # 質問リスト
    questions = [
        "Pythonはいつ開発されましたか？",
        "機械学習にはどんな種類がありますか？",
        "RAGの仕組みを説明してください。",
        "LangChainは何に使われますか？"
    ]
    
    for question in questions:
        print(f"\n❓ 質問: {question}")
        
        # 検索されるドキュメントを表示
        retrieved_docs = await retriever.ainvoke(question)
        print(f"📚 参照ドキュメント数: {len(retrieved_docs)}")
        
        # 回答生成
        answer = await rag_chain.ainvoke(question)
        print(f"💡 回答: {answer}")
        print("-" * 70)

# ===== 5. ストリーミングRAG =====
async def streaming_rag_demo(vectorstore):
    """ストリーミング出力でのRAG"""
    print("\n" + "="*70)
    print("5️⃣  ストリーミングRAG")
    print("="*70)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    template = """以下の情報を参考に、質問に詳しく答えてください。

参考情報:
{context}

質問: {question}

回答:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # ストリーミング用LLM
    streaming_llm = ChatOpenAI(
        openai_api_base="http://localhost:11434/v1",
        streaming=True,
        temperature=0.7,
        openai_api_key="EMPTY",
        model="qwen3:8b"
    )
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | streaming_llm
        | StrOutputParser()
    )
    
    question = "深層学習とTransformerについて詳しく教えてください。"
    print(f"❓ 質問: {question}\n")
    print("💡 回答（ストリーミング）:")
    print("   ", end="", flush=True)
    
    async for chunk in rag_chain.astream(question):
        print(chunk, end="", flush=True)
    
    print("\n\n✅ ストリーミング完了")

# ===== 6. MMR（Maximum Marginal Relevance）検索 =====
async def mmr_search_demo(vectorstore):
    """多様性を考慮した検索"""
    print("\n" + "="*70)
    print("6️⃣  MMR検索（多様性考慮）")
    print("="*70)
    
    query = "機械学習の技術"
    print(f"🔍 クエリ: {query}\n")
    
    # 通常の類似度検索
    print("📖 通常の類似度検索:")
    normal_docs = await vectorstore.asimilarity_search(query, k=4)
    for i, doc in enumerate(normal_docs, 1):
        content = doc.page_content.replace('\n', ' ')[:60]
        print(f"  {i}. {content}...")
    
    # MMR検索（多様性を考慮）
    print("\n📖 MMR検索（多様性重視）:")
    mmr_docs = await vectorstore.amax_marginal_relevance_search(
        query, 
        k=4,
        fetch_k=10  # 候補として10件取得し、その中から多様な4件を選択
    )
    for i, doc in enumerate(mmr_docs, 1):
        content = doc.page_content.replace('\n', ' ')[:60]
        print(f"  {i}. {content}...")
    
    print("\n💡 MMRは類似性と多様性のバランスを取った結果を返します")

# ===== メイン実行 =====
async def main():
    """全てのサンプルを実行"""
    print("\n" + "🌟"*35)
    print("LangChain RAG 完全サンプル")
    print("🌟"*35)
    
    # ベクトルストアの構築
    print("\n⚙️  初期化中...")
    vectorstore = await build_vector_store()
    
    examples = [
        ("類似ドキュメント検索", lambda: similarity_search_demo(vectorstore)),
        ("スコア付き検索", lambda: similarity_search_with_score_demo(vectorstore)),
        ("RAGチェーン：質問応答", lambda: rag_chain_demo(vectorstore)),
        ("ストリーミングRAG", lambda: streaming_rag_demo(vectorstore)),
        ("MMR検索", lambda: mmr_search_demo(vectorstore)),
    ]
    
    print("\n実行するサンプルを選択してください:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    print("0. 全て実行")
    
    try:
        choice = input("\n番号を入力 (0-5): ").strip()
        
        if choice == "0":
            for name, func in examples:
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
