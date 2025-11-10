"""
LangChain チェーン 実用例集

実際のユースケースを想定したチェーンの組み合わせパターン
"""

import asyncio
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

def get_llm(streaming: bool = False):
    return ChatOpenAI(
        openai_api_base="http://localhost:11434/v1",
        streaming=streaming,
        temperature=0.7,
        openai_api_key="EMPTY",
        model="qwen3:8b"
    )

# ===== ユースケース 1: ブログ記事自動生成システム =====
async def blog_generation_pipeline():
    """ブログ記事を自動生成する完全なパイプライン"""
    print("\n" + "="*70)
    print("📝 ユースケース1: ブログ記事自動生成システム")
    print("="*70)
    
    # ステップ1: トピック分析
    topic_analysis_prompt = ChatPromptTemplate.from_template(
        """以下のキーワードから、ターゲット読者と記事の目的を分析してください：
        
キーワード: {keywords}

以下の形式で回答してください：
- ターゲット読者: 
- 記事の目的: 
- トーン: """
    )
    
    # ステップ2: タイトル生成（3つ）
    title_generation_prompt = ChatPromptTemplate.from_template(
        """以下の情報に基づいて、魅力的なブログタイトルを3つ提案してください：

{analysis}

タイトル候補を番号付きリストで出力してください。"""
    )
    
    # ステップ3: 構成作成
    outline_prompt = ChatPromptTemplate.from_template(
        """以下のタイトルに基づいて、ブログ記事の構成（見出し）を作成してください：

{title}

以下の形式で出力：
1. 導入
2. [メインポイント1]
3. [メインポイント2]
4. [メインポイント3]
5. まとめ"""
    )
    
    # ステップ4: 各セクションの本文生成（並列実行）
    section_prompt = ChatPromptTemplate.from_template(
        """以下の見出しについて、150文字程度で内容を書いてください：

見出し: {heading}
全体のコンテキスト: {context}"""
    )
    
    # パイプライン構築
    analysis_chain = topic_analysis_prompt | get_llm() | StrOutputParser()
    title_chain = title_generation_prompt | get_llm() | StrOutputParser()
    outline_chain = outline_prompt | get_llm() | StrOutputParser()
    
    # 実行
    keywords = "AI、機械学習、Python、初心者"
    
    print(f"🔍 入力キーワード: {keywords}\n")
    
    # 分析実行
    analysis = await analysis_chain.ainvoke({"keywords": keywords})
    print(f"📊 トピック分析:\n{analysis}\n")
    
    # タイトル生成
    titles = await title_chain.ainvoke({"analysis": analysis})
    print(f"📌 タイトル候補:\n{titles}\n")
    
    # 最初のタイトルを使用
    selected_title = titles.split('\n')[0].strip()
    print(f"✅ 選択されたタイトル: {selected_title}\n")
    
    # 構成作成
    outline = await outline_chain.ainvoke({"title": selected_title})
    print(f"📋 記事構成:\n{outline}\n")
    
    return {
        "analysis": analysis,
        "title": selected_title,
        "outline": outline
    }

# ===== ユースケース 2: カスタマーサポートボット =====
async def customer_support_bot():
    """問い合わせを分類して適切に応答"""
    print("\n" + "="*70)
    print("🎧 ユースケース2: カスタマーサポートボット")
    print("="*70)
    
    # 分類プロンプト
    classify_prompt = ChatPromptTemplate.from_template(
        """以下の問い合わせを分類してください。

問い合わせ: {query}

以下のいずれかで分類してください：
- technical: 技術的な問題
- billing: 請求・支払い関連
- general: 一般的な質問

分類結果のみを回答してください。"""
    )
    
    # 応答プロンプト（カテゴリ別）
    technical_prompt = ChatPromptTemplate.from_template(
        """技術サポート担当として、以下の問題に対する解決策を提案してください：

問題: {query}

手順を明確に説明してください。"""
    )
    
    billing_prompt = ChatPromptTemplate.from_template(
        """請求サポート担当として、以下の問い合わせに丁寧に回答してください：

問い合わせ: {query}

必要な情報や手続きを説明してください。"""
    )
    
    general_prompt = ChatPromptTemplate.from_template(
        """カスタマーサポート担当として、以下の質問に親切に回答してください：

質問: {query}

わかりやすく説明してください。"""
    )
    
    # ルーター関数
    def route_query(inputs: Dict):
        category = inputs["category"].strip().lower()
        query = inputs["query"]
        
        if "technical" in category:
            return technical_prompt | get_llm() | StrOutputParser()
        elif "billing" in category:
            return billing_prompt | get_llm() | StrOutputParser()
        else:
            return general_prompt | get_llm() | StrOutputParser()
    
    # チェーン構築
    chain = (
        RunnableParallel(
            query=RunnablePassthrough(),
            category=(classify_prompt | get_llm() | StrOutputParser())
        )
        | RunnableLambda(route_query)
    )
    
    # テストケース
    queries = [
        "ログインできなくなりました。パスワードをリセットする方法を教えてください。",
        "今月の請求額が予想より高いのですが、明細を確認したいです。",
        "営業時間は何時から何時までですか？"
    ]
    
    for query in queries:
        print(f"\n❓ 問い合わせ: {query}")
        response = await chain.ainvoke(query)
        print(f"💬 回答: {response}")
        print("-" * 70)

# ===== ユースケース 3: コンテンツ品質チェッカー =====
async def content_quality_checker():
    """テキストの品質を多角的に評価"""
    print("\n" + "="*70)
    print("🔍 ユースケース3: コンテンツ品質チェッカー")
    print("="*70)
    
    # 各評価軸のプロンプト
    readability_prompt = ChatPromptTemplate.from_template(
        """以下のテキストの読みやすさを評価してください（1-10点）：

{text}

評価点と理由を簡潔に説明してください。"""
    )
    
    grammar_prompt = ChatPromptTemplate.from_template(
        """以下のテキストの文法・表現をチェックしてください：

{text}

問題点があれば指摘し、改善案を提示してください。なければ「問題なし」と回答してください。"""
    )
    
    tone_prompt = ChatPromptTemplate.from_template(
        """以下のテキストのトーン（文体）を評価してください：

{text}

フォーマル度、親しみやすさ、プロフェッショナル度を評価してください。"""
    )
    
    seo_prompt = ChatPromptTemplate.from_template(
        """以下のテキストのSEO観点での評価をしてください：

{text}

キーワードの適切さ、構造、改善提案を含めてください。"""
    )
    
    # 並列評価チェーン
    quality_chain = RunnableParallel(
        readability=(readability_prompt | get_llm() | StrOutputParser()),
        grammar=(grammar_prompt | get_llm() | StrOutputParser()),
        tone=(tone_prompt | get_llm() | StrOutputParser()),
        seo=(seo_prompt | get_llm() | StrOutputParser())
    )
    
    # テストテキスト
    text = """
    AIと機械学習は現代のビジネスに革命をもたらしています。
    企業はデータを活用して意思決定を最適化し、
    顧客体験を向上させることができるようになりました。
    Python言語を使えば、誰でも簡単にAIモデルを構築できます。
    """
    
    print(f"📄 評価対象テキスト:\n{text.strip()}\n")
    
    results = await quality_chain.ainvoke({"text": text})
    
    print("📊 評価結果:\n")
    print(f"【読みやすさ】\n{results['readability']}\n")
    print(f"【文法・表現】\n{results['grammar']}\n")
    print(f"【トーン】\n{results['tone']}\n")
    print(f"【SEO】\n{results['seo']}\n")

# ===== ユースケース 4: データ分析レポート生成 =====
async def data_analysis_report():
    """データを分析してレポートを生成"""
    print("\n" + "="*70)
    print("📈 ユースケース4: データ分析レポート生成")
    print("="*70)
    
    # データ前処理
    def preprocess_data(inputs: Dict) -> Dict:
        """データの統計情報を計算"""
        data = inputs["data"]
        
        # 簡単な統計計算
        total = sum(data)
        average = total / len(data)
        max_val = max(data)
        min_val = min(data)
        
        return {
            "raw_data": data,
            "total": total,
            "average": average,
            "max": max_val,
            "min": min_val,
            "count": len(data)
        }
    
    # 分析プロンプト（並列実行）
    trend_prompt = ChatPromptTemplate.from_template(
        """以下のデータからトレンドを分析してください：

合計: {total}
平均: {average}
最大: {max}
最小: {min}
データ点数: {count}

トレンドと傾向を説明してください。"""
    )
    
    insight_prompt = ChatPromptTemplate.from_template(
        """以下の統計情報から、ビジネス上の洞察を提供してください：

合計: {total}
平均: {average}
最大: {max}
最小: {min}

アクションアイテムを含めてください。"""
    )
    
    summary_prompt = ChatPromptTemplate.from_template(
        """以下の分析結果をエグゼクティブサマリーとしてまとめてください：

トレンド分析:
{trend}

ビジネス洞察:
{insight}

3行程度で要約してください。"""
    )
    
    # チェーン構築
    chain = (
        RunnableLambda(preprocess_data)
        | RunnableParallel(
            stats=RunnablePassthrough(),
            trend=(trend_prompt | get_llm() | StrOutputParser()),
            insight=(insight_prompt | get_llm() | StrOutputParser())
        )
        | RunnableParallel(
            trend=lambda x: x["trend"],
            insight=lambda x: x["insight"],
            summary=(
                RunnableLambda(lambda x: {"trend": x["trend"], "insight": x["insight"]})
                | summary_prompt
                | get_llm()
                | StrOutputParser()
            )
        )
    )
    
    # サンプルデータ（月次売上）
    sales_data = [120, 135, 150, 145, 160, 175, 180, 195, 210, 205, 225, 240]
    
    print(f"📊 売上データ（月次）: {sales_data}\n")
    
    result = await chain.ainvoke({"data": sales_data})
    
    print(f"📈 トレンド分析:\n{result['trend']}\n")
    print(f"💡 ビジネス洞察:\n{result['insight']}\n")
    print(f"📋 エグゼクティブサマリー:\n{result['summary']}\n")

# ===== メイン実行 =====
async def main():
    """全ての実用例を実行"""
    print("\n" + "🌟"*35)
    print("LangChain チェーン 実用例集")
    print("🌟"*35)
    
    use_cases = [
        ("ブログ記事自動生成", blog_generation_pipeline),
        ("カスタマーサポートボット", customer_support_bot),
        ("コンテンツ品質チェッカー", content_quality_checker),
        ("データ分析レポート生成", data_analysis_report),
    ]
    
    print("\n実行するユースケースを選択してください:")
    for i, (name, _) in enumerate(use_cases, 1):
        print(f"{i}. {name}")
    print("0. 全て実行")
    
    try:
        choice = input("\n番号を入力 (0-4): ").strip()
        
        if choice == "0":
            for name, func in use_cases:
                await func()
        elif choice.isdigit() and 1 <= int(choice) <= len(use_cases):
            _, func = use_cases[int(choice) - 1]
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
    print("実用例の実行完了！")
    print("🎉"*35 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
