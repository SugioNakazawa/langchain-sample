"""
LangChain チェーンアーキテクチャ サンプル集

このファイルでは、LangChainの様々なチェーンパターンを実装します：
1. Simple Chain - 基本的な単一チェーン
2. Sequential Chain - 複数ステップの順次実行
3. Router Chain - 条件分岐
4. Transform Chain - データ変換
5. LCEL (LangChain Expression Language) - 最新の推奨方法
6. RAG Chain - 検索拡張生成
"""

import asyncio
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field

# ===== 共通設定 =====
def get_llm(streaming: bool = False, temperature: float = 0.7):
    """LLMインスタンスを取得"""
    return ChatOpenAI(
        openai_api_base="http://localhost:11434/v1",
        streaming=streaming,
        temperature=temperature,
        openai_api_key="EMPTY",
        model="qwen3:8b"
    )

# ===== 1. Simple Chain - 基本的な単一チェーン =====
async def simple_chain_example():
    """最もシンプルなチェーンの例"""
    print("\n" + "="*60)
    print("1️⃣  Simple Chain Example")
    print("="*60)
    
    # プロンプトテンプレート
    prompt = ChatPromptTemplate.from_template(
        "あなたはプロの翻訳者です。以下の文章を{target_lang}に翻訳してください：\n\n{text}"
    )
    question = "今日は良い天気ですね。散歩に行きましょう。"
    
    # チェーン構築 (LCEL方式)
    chain = prompt | get_llm() | StrOutputParser()
    
    # 実行
    result = await chain.ainvoke({
        "target_lang": "英語",
        "text": question
    })
    print(f"\n質問: {question}")
    print(f"\n翻訳結果: {result}")
    return result

# ===== 2. Sequential Chain - 複数ステップの順次実行 =====
async def sequential_chain_example():
    """複数のチェーンを順次実行する例"""
    print("\n" + "="*60)
    print("2️⃣  Sequential Chain Example")
    print("="*60)
    
    # ステップ1: トピック生成
    topic_prompt = ChatPromptTemplate.from_template(
        "以下のキーワードから、ブログ記事のタイトルを1つ提案してください：{keywords}"
    )
    
    # ステップ2: 概要作成
    outline_prompt = ChatPromptTemplate.from_template(
        "以下のブログタイトルに基づいて、3つの見出しを箇条書きで作成してください：\n\n{title}"
    )
    
    # ステップ3: 本文作成
    content_prompt = ChatPromptTemplate.from_template(
        "以下のタイトルと見出しに基づいて、200文字程度の導入文を書いてください：\n\nタイトル: {title}\n見出し:\n{outline}"
    )
    
    # LCEL で順次実行チェーンを構築
    chain = (
        {"keywords": RunnablePassthrough()}
        | RunnableParallel(
            title=(topic_prompt | get_llm() | StrOutputParser()),
            keywords=RunnablePassthrough()
        )
        | RunnableParallel(
            title=lambda x: x["title"],
            outline=(
                RunnableLambda(lambda x: {"title": x["title"]})
                | outline_prompt
                | get_llm()
                | StrOutputParser()
            )
        )
        | RunnableParallel(
            title=lambda x: x["title"],
            outline=lambda x: x["outline"],
            content=(
                RunnableLambda(lambda x: {"title": x["title"], "outline": x["outline"]})
                | content_prompt
                | get_llm()
                | StrOutputParser()
            )
        )
    )
    
    result = await chain.ainvoke("AI, 機械学習, Python")
    
    print(f"\n📝 タイトル:\n{result['title']}")
    print(f"\n📋 見出し:\n{result['outline']}")
    print(f"\n✍️  導入文:\n{result['content']}")
    
    return result

# ===== 3. Router Chain - 条件分岐 =====
async def router_chain_example():
    """入力内容に応じて処理を分岐する例"""
    print("\n" + "="*60)
    print("3️⃣  Router Chain Example")
    print("="*60)
    
    # 分類プロンプト
    classifier_prompt = ChatPromptTemplate.from_template(
        """以下の質問を分類してください。「技術」「ビジネス」「日常」のいずれかで答えてください。
        
質問: {question}

分類:"""
    )
    
    # カテゴリ別の応答プロンプト
    tech_prompt = ChatPromptTemplate.from_template(
        "技術的な観点から詳しく説明します：{question}"
    )
    
    business_prompt = ChatPromptTemplate.from_template(
        "ビジネスの視点で実用的に説明します：{question}"
    )
    
    casual_prompt = ChatPromptTemplate.from_template(
        "分かりやすく日常的な言葉で説明します：{question}"
    )
    
    # ルーター関数
    def route_question(input_dict: Dict) -> Any:
        category = input_dict["category"].strip().lower()
        question = input_dict["question"]
        
        if "技術" in category:
            return tech_prompt | get_llm() | StrOutputParser()
        elif "ビジネス" in category:
            return business_prompt | get_llm() | StrOutputParser()
        else:
            return casual_prompt | get_llm() | StrOutputParser()
    
    # ルーターチェーン構築
    chain = (
        RunnableParallel(
            question=RunnablePassthrough(),
            category=(classifier_prompt | get_llm() | StrOutputParser())
        )
        | RunnableLambda(route_question)
    )
    
    # テストケース
    questions = [
        "Pythonの非同期プログラミングとは？",
        "スタートアップの資金調達方法は？",
        "朝ごはんは何を食べるのが良い？"
    ]
    
    for q in questions:
        result = await chain.ainvoke(q)
        print(f"\n❓ 質問: {q}")
        print(f"💡 回答: {result}\n")
        print("-" * 60)

# ===== 4. Transform Chain - データ変換 =====
async def transform_chain_example():
    """データ変換を含むチェーンの例"""
    print("\n" + "="*60)
    print("4️⃣  Transform Chain Example")
    print("="*60)
    
    # テキスト前処理関数
    def preprocess_text(inputs: Dict) -> Dict:
        """テキストのクリーニングと正規化"""
        text = inputs["text"]
        # 小文字化、空白の正規化
        cleaned = " ".join(text.lower().split())
        word_count = len(cleaned.split())
        
        return {
            "original": text,
            "cleaned": cleaned,
            "word_count": word_count
        }
    
    # 後処理関数
    def postprocess_result(inputs: Dict) -> Dict:
        """結果の整形"""
        return {
            "summary": inputs["result"],
            "metadata": {
                "original_length": len(inputs["original"]),
                "word_count": inputs["word_count"]
            }
        }
    
    # プロンプト
    summarize_prompt = ChatPromptTemplate.from_template(
        "以下のテキストを30文字以内で要約してください：\n\n{cleaned}"
    )
    
    # チェーン構築
    chain = (
        RunnableLambda(preprocess_text)
        | RunnableParallel(
            original=lambda x: x["original"],
            cleaned=lambda x: x["cleaned"],
            word_count=lambda x: x["word_count"],
            result=(
                RunnableLambda(lambda x: {"cleaned": x["cleaned"]})
                | summarize_prompt
                | get_llm()
                | StrOutputParser()
            )
        )
        | RunnableLambda(postprocess_result)
    )
    
    # 実行
    text = """
    人工知能（AI）は、コンピュータサイエンスの一分野で、
    人間の知能を模倣するシステムの研究開発を行います。
    機械学習やディープラーニングなどの技術が含まれます。
    """
    
    result = await chain.ainvoke({"text": text})
    
    print(f"📄 元のテキスト長: {result['metadata']['original_length']} 文字")
    print(f"📊 単語数: {result['metadata']['word_count']}")
    print(f"📝 要約: {result['summary']}")
    
    return result

# ===== 5. LCEL Parallel Chain - 並列実行 =====
class AnalysisResult(BaseModel):
    """分析結果の構造化データ"""
    sentiment: str = Field(description="感情分析結果（positive/negative/neutral）")
    category: str = Field(description="カテゴリ分類")
    keywords: List[str] = Field(description="キーワードリスト")

async def parallel_chain_example():
    """複数のタスクを並列実行する例"""
    print("\n" + "="*60)
    print("5️⃣  Parallel Chain Example (LCEL)")
    print("="*60)
    
    # 各分析用のプロンプト
    sentiment_prompt = ChatPromptTemplate.from_template(
        "以下のテキストの感情を分析し、positive/negative/neutralのいずれかで答えてください：\n\n{text}"
    )
    
    category_prompt = ChatPromptTemplate.from_template(
        "以下のテキストのカテゴリを「技術」「ビジネス」「エンタメ」「その他」から選んでください：\n\n{text}"
    )
    
    keywords_prompt = ChatPromptTemplate.from_template(
        "以下のテキストから重要なキーワードを3つ抽出し、カンマ区切りで列挙してください：\n\n{text}"
    )
    
    summary_prompt = ChatPromptTemplate.from_template(
        "以下のテキストを一行で要約してください：\n\n{text}"
    )
    
    # 並列実行チェーン
    parallel_chain = RunnableParallel(
        sentiment=(sentiment_prompt | get_llm() | StrOutputParser()),
        category=(category_prompt | get_llm() | StrOutputParser()),
        keywords=(keywords_prompt | get_llm() | StrOutputParser()),
        summary=(summary_prompt | get_llm() | StrOutputParser()),
        original=lambda x: x["text"]
    )
    
    # 実行
    text = """
    新しいAI技術により、プログラミングの効率が大幅に向上しました。
    開発者はより創造的な作業に集中できるようになり、
    生産性が飛躍的に改善されています。素晴らしい進歩です！
    """
    
    result = await parallel_chain.ainvoke({"text": text})
    
    print(f"📊 感情分析: {result['sentiment']}")
    print(f"🏷️  カテゴリ: {result['category']}")
    print(f"🔑 キーワード: {result['keywords']}")
    print(f"📝 要約: {result['summary']}")
    
    return result

# ===== 6. Streaming Chain - ストリーミング出力 =====
async def streaming_chain_example():
    """ストリーミング出力のチェーン例"""
    print("\n" + "="*60)
    print("6️⃣  Streaming Chain Example")
    print("="*60)
    
    prompt = ChatPromptTemplate.from_template(
        "以下のトピックについて、300文字程度で説明してください：{topic}"
    )
    
    chain = prompt | get_llm(streaming=True) | StrOutputParser()
    
    print("🎬 ストリーミング開始...\n")
    print("📝 ", end="", flush=True)
    
    full_response = ""
    async for chunk in chain.astream({"topic": "量子コンピュータの未来"}):
        print(chunk, end="", flush=True)
        full_response += chunk
    
    print("\n\n✅ ストリーミング完了")
    return full_response

# ===== 7. Memory Chain - 会話履歴を持つチェーン =====
async def memory_chain_example():
    """会話履歴を管理するチェーン例（LCEL版）"""
    print("\n" + "="*60)
    print("7️⃣  Memory Chain Example (Conversation)")
    print("="*60)
    
    # 会話履歴を保持するリスト
    conversation_history = []
    
    # プロンプトテンプレート（履歴を含む）
    prompt = ChatPromptTemplate.from_template(
        """以下は今までの会話履歴です：
{history}

ユーザー: {input}
AI:"""
    )
    
    # チェーン構築
    chain = prompt | get_llm() | StrOutputParser()
    
    # 会話シミュレーション
    conversations = [
        "こんにちは！私の名前は太郎です。",
        "私の好きなプログラミング言語はPythonです。",
        "私の名前を覚えていますか？",
        "私の好きな言語は何でしたっけ？"
    ]
    
    for user_input in conversations:
        # 履歴を文字列に変換
        history_text = "\n".join([
            f"ユーザー: {h['user']}\nAI: {h['ai']}"
            for h in conversation_history
        ]) if conversation_history else "（まだ会話履歴はありません）"
        
        # AIの応答を取得
        response = await chain.ainvoke({
            "history": history_text,
            "input": user_input
        })
        
        # 履歴に追加
        conversation_history.append({
            "user": user_input,
            "ai": response
        })
        
        print(f"👤 ユーザー: {user_input}")
        print(f"🤖 AI: {response}\n")
        print("-" * 60)

# ===== 8. Custom Chain - カスタムチェーン =====
async def custom_chain_example():
    """カスタムロジックを含むチェーン例"""
    print("\n" + "="*60)
    print("8️⃣  Custom Chain Example")
    print("="*60)
    
    # カスタム処理関数
    async def validate_and_enhance(inputs: Dict) -> Dict:
        """入力を検証して拡張"""
        query = inputs["query"]
        
        # 簡単なバリデーション
        if len(query) < 5:
            return {
                "error": "クエリが短すぎます",
                "enhanced_query": None
            }
        
        # クエリの拡張
        enhanced = f"{query}（具体例や実用的な情報を含めて詳しく）"
        
        return {
            "error": None,
            "enhanced_query": enhanced,
            "original_query": query
        }
    
    # 条件付き実行
    def execute_if_valid(inputs: Dict) -> Any:
        if inputs["error"]:
            return RunnableLambda(lambda x: f"エラー: {x['error']}")
        else:
            prompt = ChatPromptTemplate.from_template("{enhanced_query}")
            return prompt | get_llm() | StrOutputParser()
    
    # カスタムチェーン
    chain = (
        RunnableLambda(validate_and_enhance)
        | RunnableLambda(execute_if_valid)
    )
    
    # テスト
    test_queries = ["AI", "機械学習とディープラーニングの違い"]
    
    for query in test_queries:
        print(f"\n❓ クエリ: {query}")
        result = await chain.ainvoke({"query": query})
        print(f"💡 結果: {result}")
        print("-" * 60)

# ===== メイン実行 =====
async def main():
    """全てのサンプルを実行"""
    print("\n" + "🌟"*30)
    print("LangChain チェーンアーキテクチャ サンプル集")
    print("🌟"*30)
    
    examples = [
        ("Simple Chain", simple_chain_example),
        ("Sequential Chain", sequential_chain_example),
        ("Router Chain", router_chain_example),
        ("Transform Chain", transform_chain_example),
        ("Parallel Chain", parallel_chain_example),
        ("Streaming Chain", streaming_chain_example),
        ("Memory Chain", memory_chain_example),
        ("Custom Chain", custom_chain_example),
    ]
    
    print("\n実行するサンプルを選択してください:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    print("0. 全て実行")
    
    try:
        choice = input("\n番号を入力 (0-8): ").strip()
        
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
    
    print("\n" + "🎉"*30)
    print("サンプル実行完了！")
    print("🎉"*30 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
