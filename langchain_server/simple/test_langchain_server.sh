#!/bin/bash

# LangChain Server テストスクリプト
# 使用方法: ./test_langchain_server.sh

set -e

# カラー出力の設定
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SERVER_URL="http://localhost:8000"

echo -e "${BLUE}=== LangChain Server テスト ===${NC}"

# サーバーの状態確認
echo -e "${YELLOW}サーバーの状態を確認中...${NC}"
if ! curl -s "$SERVER_URL/v1/models" > /dev/null; then
    echo -e "${RED}エラー: サーバーに接続できません ($SERVER_URL)${NC}"
    echo -e "${YELLOW}サーバーが起動していることを確認してください${NC}"
    exit 1
fi

echo -e "${GREEN}サーバーが利用可能です${NC}"

# モデル一覧の取得
echo -e "${YELLOW}利用可能なモデルを取得中...${NC}"
curl -s "$SERVER_URL/v1/models" | jq .

echo ""

# チャット完了のテスト
echo -e "${YELLOW}チャット完了APIをテスト中...${NC}"
RESPONSE=$(curl -s -X POST "$SERVER_URL/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "nautilus-llm",
       "messages": [
         {"role": "user", "content": "こんにちは！簡単に自己紹介してください。"}
       ]
     }')

echo -e "${GREEN}レスポンス:${NC}"
echo "$RESPONSE" | jq .

echo ""
echo -e "${GREEN}テスト完了！${NC}"