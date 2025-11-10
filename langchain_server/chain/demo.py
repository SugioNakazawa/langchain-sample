#!/usr/bin/env python3
"""
LangChain チェーンアーキテクチャ - クイックスタートガイド

各チェーンパターンの簡単な使用例
"""

import asyncio
from chain import (
    simple_chain_example,
    sequential_chain_example,
    router_chain_example,
    transform_chain_example,
    parallel_chain_example,
    streaming_chain_example,
    memory_chain_example,
    custom_chain_example
)

async def demo():
    """デモ実行"""
    print("\n" + "="*70)
    print("🚀 LangChain チェーンアーキテクチャ - クイックデモ")
    print("="*70)
    
    # 1. Simple Chain
    print("\n【1】Simple Chain - 基本的な翻訳")
    await simple_chain_example()
    
    # 2. Parallel Chain  
    print("\n【2】Parallel Chain - テキスト分析（感情・カテゴリ・キーワード）")
    await parallel_chain_example()
    
    # 3. Streaming Chain
    print("\n【3】Streaming Chain - リアルタイム生成")
    await streaming_chain_example()
    
    print("\n" + "="*70)
    print("✅ デモ完了！詳細は各チェーンの実行結果を確認してください。")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(demo())
