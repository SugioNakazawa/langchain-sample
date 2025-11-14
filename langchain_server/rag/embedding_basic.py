"""
LangChain Embedding 基本サンプル

このスクリプトでは、以下の基本的なembedding操作を実演します：
1. テキストのベクトル化
2. 類似度検索
3. 簡単なセマンティック検索
"""

import asyncio
from typing import List
import numpy as np
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

# ===== 設定 =====
def get_embeddings():
    """Embedding モデルを取得"""
    return OllamaEmbeddings(
        base_url="http://localhost:11434",
        model="mxbai-embed-large"  # Ollamaの軽量embedgingモデル
    )

# ===== 1. 基本的なテキストのベクトル化 =====
async def basic_embedding_example():
    """基本的なembedding生成の例"""
    print("\n" + "="*70)
    print("1️⃣  基本的なテキストのベクトル化")
    print("="*70)
    
    embeddings = get_embeddings()
    
    # 単一テキストのembedding
    text = "人工知能と機械学習は現代技術の重要な分野です。"
    
    print(f"📝 元のテキスト: {text}")
    
    # ベクトル化
    vector = await embeddings.aembed_query(text)
    
    print(f"📊 ベクトルの次元数: {len(vector)}")
    print(f"📈 ベクトルの最初の10要素: {vector[:10]}")
    print(f"✅ ベクトル化完了")
    
    return vector

# ===== 2. 複数テキストのバッチ処理 =====
async def batch_embedding_example():
    """複数テキストのembedding生成"""
    print("\n" + "="*70)
    print("2️⃣  複数テキストのバッチembedding")
    print("="*70)
    
    embeddings = get_embeddings()
    
    # 複数のテキスト
    texts = [
        "Pythonはプログラミング言語です。",
        "機械学習にはデータが必要です。",
        "深層学習はニューラルネットワークを使用します。",
        "自然言語処理はテキストを扱います。"
    ]
    
    print("📝 テキストリスト:")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")
    
    # バッチでベクトル化
    vectors = await embeddings.aembed_documents(texts)
    
    print(f"\n📊 生成されたベクトル数: {len(vectors)}")
    print(f"📏 各ベクトルの次元: {len(vectors[0])}")
    print("✅ バッチ処理完了")
    
    return texts, vectors

# ===== 3. コサイン類似度計算 =====
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """2つのベクトル間のコサイン類似度を計算"""
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)
    
    return dot_product / (norm1 * norm2)

async def similarity_search_example():
    """類似度検索の例"""
    print("\n" + "="*70)
    print("3️⃣  類似度検索")
    print("="*70)
    
    embeddings = get_embeddings()
    
    # ドキュメントコレクション
    documents = [
        "猫は可愛いペットです。",
        "犬は忠実な友達です。",
        "Pythonはプログラミング言語です。",
        "機械学習はAIの一分野です。",
        "鳥は空を飛びます。",
        "魚は水中で生活します。",
        "自然言語処理はテキスト分析に使われます。",
        "深層学習はニューラルネットワークを活用します。"
    ]
    
    # クエリ
    query = "ペットについて教えて"
    
    print(f"🔍 クエリ: {query}\n")
    print("📚 ドキュメントコレクション:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")
    
    # クエリとドキュメントをベクトル化
    query_vector = await embeddings.aembed_query(query)
    doc_vectors = await embeddings.aembed_documents(documents)
    
    # 類似度を計算
    similarities = []
    for i, doc_vector in enumerate(doc_vectors):
        similarity = cosine_similarity(query_vector, doc_vector)
        similarities.append((i, documents[i], similarity))
    
    # 類似度でソート
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    print("\n🎯 類似度ランキング（上位5件）:")
    for rank, (idx, doc, score) in enumerate(similarities[:5], 1):
        print(f"  {rank}位 (類似度: {score:.4f}): {doc}")
    
    return similarities

# ===== 4. セマンティック検索 =====
async def semantic_search_example():
    """意味的に類似した文章を検索"""
    print("\n" + "="*70)
    print("4️⃣  セマンティック検索")
    print("="*70)
    
    embeddings = get_embeddings()
    
    # 技術記事のタイトルコレクション
    articles = [
        "Pythonで始める機械学習入門",
        "深層学習の基礎：ニューラルネットワークを理解する",
        "自然言語処理の最新技術トレンド",
        "データサイエンスのためのPandas活用術",
        "クラウドコンピューティングの基本概念",
        "Dockerコンテナで開発環境を構築する方法",
        "RESTful API設計のベストプラクティス",
        "機械学習モデルのデプロイメント戦略",
        "TransformerアーキテクチャとBERTの解説",
        "時系列データ分析とLSTMネットワーク"
    ]
    
    queries = [
        "AIについて学びたい",
        "Pythonの使い方を知りたい",
        "インフラの構築方法"
    ]
    
    print("📚 記事タイトルコレクション:")
    for i, article in enumerate(articles, 1):
        print(f"  {i:2d}. {article}")
    
    # 記事をベクトル化
    article_vectors = await embeddings.aembed_documents(articles)
    
    for query in queries:
        print(f"\n🔍 クエリ: 「{query}」")
        
        # クエリをベクトル化
        query_vector = await embeddings.aembed_query(query)
        
        # 類似度計算
        results = []
        for i, article_vector in enumerate(article_vectors):
            similarity = cosine_similarity(query_vector, article_vector)
            results.append((articles[i], similarity))
        
        # ソートして上位3件表示
        results.sort(key=lambda x: x[1], reverse=True)
        
        print("  📖 関連記事（上位3件）:")
        for rank, (article, score) in enumerate(results[:3], 1):
            print(f"    {rank}. {article} (類似度: {score:.4f})")

# ===== 5. 多言語embedding =====
async def multilingual_embedding_example():
    """多言語テキストのembedding"""
    print("\n" + "="*70)
    print("5️⃣  多言語embedding")
    print("="*70)
    
    embeddings = get_embeddings()
    
    # 同じ意味の異なる言語のテキスト
    texts = {
        "日本語": "こんにちは、今日は良い天気ですね。",
        "英語": "Hello, it's a nice day today.",
        "関連": "天気が良くて気持ちいいです。",
        "非関連": "プログラミングは楽しいです。"
    }
    
    print("📝 テキストサンプル:")
    for lang, text in texts.items():
        print(f"  {lang}: {text}")
    
    # ベクトル化
    vectors = {}
    for lang, text in texts.items():
        vectors[lang] = await embeddings.aembed_query(text)
    
    # 類似度マトリックス
    print("\n📊 類似度マトリックス:")
    print(f"{'':12}", end="")
    for lang in texts.keys():
        print(f"{lang:12}", end="")
    print()
    
    for lang1 in texts.keys():
        print(f"{lang1:12}", end="")
        for lang2 in texts.keys():
            similarity = cosine_similarity(vectors[lang1], vectors[lang2])
            print(f"{similarity:12.4f}", end="")
        print()
    
    print("\n💡 観察:")
    print("  - 日本語と英語（同じ意味）の類似度が高い")
    print("  - 関連テキストも比較的高い類似度")
    print("  - 非関連テキストは類似度が低い")

# ===== メイン実行 =====
async def main():
    """全てのサンプルを実行"""
    print("\n" + "🌟"*35)
    print("LangChain Embedding 基本サンプル集")
    print("🌟"*35)
    
    examples = [
        ("基本的なテキストのベクトル化", basic_embedding_example),
        ("複数テキストのバッチembedding", batch_embedding_example),
        ("類似度検索", similarity_search_example),
        ("セマンティック検索", semantic_search_example),
        ("多言語embedding", multilingual_embedding_example),
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
