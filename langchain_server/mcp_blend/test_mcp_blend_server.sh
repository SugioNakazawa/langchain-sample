#!/bin/bash

# MCP Blend Server テストスクリプト
# 使用方法: ./test_mcp_blend_server.sh

set -e

# カラー出力の設定
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

MCP_URL="http://localhost:9100"
APP_URL="http://localhost:8000"

echo -e "${BLUE}=== MCP Blend Server テスト ===${NC}"

# MCPサーバーの状態確認
echo -e "${YELLOW}MCPサーバーの状態を確認中...${NC}"
if ! curl -s "$MCP_URL" -X POST -H "Content-Type: application/json" -d '{"tool_id":"test"}' > /dev/null; then
    echo -e "${RED}エラー: MCPサーバーに接続できません ($MCP_URL)${NC}"
    echo -e "${YELLOW}MCPサーバーが起動していることを確認してください${NC}"
    exit 1
fi

echo -e "${GREEN}MCPサーバーが利用可能です${NC}"

# ブレンド最適化のテスト
echo -e "${YELLOW}ブレンド最適化APIをテスト中...${NC}"
RESPONSE=$(curl -s -X POST "$MCP_URL" \
     -H "Content-Type: application/json" \
     -d '{
       "tool_id": "optimize_blend",
       "input": {
         "oils": [
           {"name": "Soybean", "cost": 1.1, "iodine": 120},
           {"name": "Palm", "cost": 0.9, "iodine": 80},
           {"name": "Rapeseed", "cost": 1.0, "iodine": 110}
         ],
         "demand": 1000
       }
     }')

echo -e "${GREEN}MCPサーバーレスポンス:${NC}"
echo "$RESPONSE" | jq .

echo ""

# Appサーバーの確認（起動している場合）
echo -e "${YELLOW}Appサーバーの状態を確認中...${NC}"
if curl -s "$APP_URL/v1/models" > /dev/null; then
    echo -e "${GREEN}Appサーバーが利用可能です${NC}"
    
    # モデル一覧の取得
    echo -e "${YELLOW}利用可能なモデルを取得中...${NC}"
    curl -s "$APP_URL/v1/models" | jq .
    
    echo ""
    
    # チャット完了のテスト
    echo -e "${YELLOW}チャット完了APIをテスト中...${NC}"
    CHAT_RESPONSE=$(curl -s -X POST "$APP_URL/v1/chat/completions" \
         -H "Content-Type: application/json" \
         -d '{
           "model": "nautilus-llm",
           "messages": [
             {"role": "user", "content": "最適ブレンドを計算してください"}
           ]
         }')
    
    echo -e "${GREEN}Appサーバーレスポンス:${NC}"
    echo "$CHAT_RESPONSE" | jq .
else
    echo -e "${YELLOW}Appサーバーは起動していません${NC}"
fi

echo ""

# パフォーマンステスト
echo -e "${YELLOW}パフォーマンステストを実行中...${NC}"
echo -e "${BLUE}複数の最適化リクエストを並行実行...${NC}"

for i in {1..3}; do
    (
        curl -s -X POST "$MCP_URL" \
             -H "Content-Type: application/json" \
             -d "{
               \"tool_id\": \"optimize_blend\",
               \"input\": {
                 \"oils\": [
                   {\"name\": \"Oil_$i\", \"cost\": $(echo "1.0 + 0.1 * $i" | bc), \"iodine\": $((100 + i * 10))},
                   {\"name\": \"Oil_$(($i + 1))\", \"cost\": $(echo "0.9 + 0.1 * $i" | bc), \"iodine\": $((90 + i * 5))}
                 ],
                 \"demand\": $((1000 + i * 100))
               }
             }" > /dev/null
        echo -e "${GREEN}リクエスト $i 完了${NC}"
    ) &
done

wait

echo -e "${GREEN}パフォーマンステスト完了！${NC}"
echo ""
echo -e "${GREEN}全テスト完了！${NC}"