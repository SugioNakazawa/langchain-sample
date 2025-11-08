import asyncio
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class Joke(BaseModel):
    id: int = Field(..., description="Unique identifier for the joke")
    setup: str = Field(..., description="The setup or question part of the joke")
    punchline: str = Field(..., description="The punchline or answer part of the joke")

async def stream_structured_output():
    """ストリーミングで構造化出力を受信"""
    llm = ChatOpenAI(
        openai_api_base="http://localhost:11434/v1",
        streaming=True,  # ストリーミング有効化
        temperature=1,
        openai_api_key="EMPTY",
        model="gpt-oss:20b"
    )
    
    # 構造化出力の設定
    structured_llm = llm.with_structured_output(Joke)
    
    print("🎭 猫に関するジョークを生成中...")
    print("=" * 50)
    
    try:
        # ストリーミングで結果を受信
        async for chunk in structured_llm.astream("猫に関するジョークを教えてください。"):
            # チャンクが完全なJokeオブジェクトの場合
            if isinstance(chunk, Joke):
                print("\n✅ 完成したジョーク:")
                print(f"ID: {chunk.id}")
                print(f"セットアップ: {chunk.setup}")
                print(f"オチ: {chunk.punchline}")
                print("=" * 50)
            else:
                # 部分的な結果やメタデータの場合
                print(f"📝 受信中: {chunk}")
                
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")

async def stream_regular_output():
    """通常のテキスト出力をストリーミング受信（比較用）"""
    llm = ChatOpenAI(
        openai_api_base="http://localhost:11434/v1",
        streaming=True,
        temperature=1,
        openai_api_key="EMPTY",
        model="qwen3:4b"
    )
    
    print("\n🔄 通常のストリーミング出力（比較用）:")
    print("=" * 50)
    
    try:
        async for chunk in llm.astream("猫に関するジョークを教えてください。"):
            # AIMessageチャンクからコンテンツを抽出
            if hasattr(chunk, 'content') and chunk.content:
                print(chunk.content, end='', flush=True)
        print("\n" + "=" * 50)
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")

async def main():
    """メイン実行関数"""
    print("🚀 LangChain ストリーミング構造化出力デモ\n")
    
    # 1. 構造化出力のストリーミング
    await stream_structured_output()
    
    # 2. 通常出力のストリーミング（比較用）
    await stream_regular_output()

if __name__ == "__main__":
    # 非同期関数を実行
    asyncio.run(main())
