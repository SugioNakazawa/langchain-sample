#!/bin/bash

# MCP Blend Server 実行スクリプト
# 使用方法: ./run_mcp_blend_server.sh

set -e

# カラー出力の設定
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 設定
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
VENV_PATH="$PROJECT_ROOT/venv"
MCP_SERVER_FILE="$SCRIPT_DIR/mcp_blend_server.py"
APP_SERVER_FILE="$SCRIPT_DIR/mcp_app_server.py"
MCP_PORT="9100"
APP_PORT="8000"

echo -e "${BLUE}=== MCP Blend Server 起動スクリプト ===${NC}"
echo -e "${YELLOW}プロジェクトディレクトリ: $PROJECT_ROOT${NC}"
echo -e "${YELLOW}仮想環境: $VENV_PATH${NC}"
echo -e "${YELLOW}MCPサーバーファイル: $MCP_SERVER_FILE${NC}"
echo -e "${YELLOW}Appサーバーファイル: $APP_SERVER_FILE${NC}"

# 仮想環境の確認
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}エラー: 仮想環境が見つかりません: $VENV_PATH${NC}"
    echo -e "${YELLOW}以下のコマンドで仮想環境を作成してください:${NC}"
    echo "  cd $PROJECT_ROOT"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install fastapi uvicorn langchain langchain-openai pulp requests"
    exit 1
fi

# サーバーファイルの確認
if [ ! -f "$MCP_SERVER_FILE" ]; then
    echo -e "${RED}エラー: MCPサーバーファイルが見つかりません: $MCP_SERVER_FILE${NC}"
    exit 1
fi

if [ ! -f "$APP_SERVER_FILE" ]; then
    echo -e "${RED}エラー: Appサーバーファイルが見つかりません: $APP_SERVER_FILE${NC}"
    exit 1
fi

# 仮想環境のアクティベート
echo -e "${GREEN}仮想環境をアクティベートしています...${NC}"
source "$VENV_PATH/bin/activate"

# Ollamaサーバーの確認
echo -e "${GREEN}Ollamaサーバーの確認中...${NC}"
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}警告: Ollamaサーバーに接続できません (http://localhost:11434)${NC}"
    echo -e "${YELLOW}Ollamaを起動してください: ollama serve${NC}"
    echo -e "${YELLOW}続行しますか? (y/N)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}Ollamaサーバーが利用可能です${NC}"
fi

# ポートの確認
if lsof -i :$MCP_PORT > /dev/null 2>&1; then
    echo -e "${YELLOW}警告: ポート $MCP_PORT は既に使用されています${NC}"
    echo -e "${YELLOW}続行しますか? (y/N)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

if lsof -i :$APP_PORT > /dev/null 2>&1; then
    echo -e "${YELLOW}警告: ポート $APP_PORT は既に使用されています${NC}"
    echo -e "${YELLOW}続行しますか? (y/N)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 起動方法の選択
echo -e "${BLUE}起動方法を選択してください:${NC}"
echo -e "${YELLOW}1) MCPサーバーのみ起動 (ポート $MCP_PORT)${NC}"
echo -e "${YELLOW}2) Appサーバーのみ起動 (ポート $APP_PORT)${NC}"
echo -e "${YELLOW}3) 両方のサーバーを起動 (MCP: $MCP_PORT, App: $APP_PORT)${NC}"
echo -e "${YELLOW}4) テスト実行${NC}"
read -p "選択 (1-4): " choice

case $choice in
    1)
        echo -e "${GREEN}MCPサーバーを起動しています...${NC}"
        echo -e "${BLUE}URL: http://0.0.0.0:$MCP_PORT${NC}"
        echo -e "${YELLOW}停止するには Ctrl+C を押してください${NC}"
        cd "$PROJECT_ROOT"
        python "$MCP_SERVER_FILE"
        ;;
    2)
        echo -e "${GREEN}Appサーバーを起動しています...${NC}"
        echo -e "${BLUE}URL: http://0.0.0.0:$APP_PORT${NC}"
        echo -e "${BLUE}API: http://0.0.0.0:$APP_PORT/v1/chat/completions${NC}"
        echo -e "${BLUE}Models: http://0.0.0.0:$APP_PORT/v1/models${NC}"
        echo -e "${YELLOW}停止するには Ctrl+C を押してください${NC}"
        cd "$PROJECT_ROOT"
        python "$APP_SERVER_FILE"
        ;;
    3)
        echo -e "${GREEN}両方のサーバーを起動しています...${NC}"
        cd "$PROJECT_ROOT"
        
        # MCPサーバーをバックグラウンドで起動
        echo -e "${BLUE}MCPサーバーをバックグラウンドで起動中...${NC}"
        nohup python "$MCP_SERVER_FILE" > mcp_server.log 2>&1 &
        MCP_PID=$!
        echo -e "${GREEN}MCPサーバー起動完了 (PID: $MCP_PID, ポート: $MCP_PORT)${NC}"
        
        # 少し待ってからAppサーバーを起動
        sleep 2
        echo -e "${BLUE}Appサーバーを起動中...${NC}"
        echo -e "${BLUE}URL: http://0.0.0.0:$APP_PORT${NC}"
        echo -e "${BLUE}API: http://0.0.0.0:$APP_PORT/v1/chat/completions${NC}"
        echo -e "${BLUE}Models: http://0.0.0.0:$APP_PORT/v1/models${NC}"
        echo -e "${YELLOW}停止するには Ctrl+C を押してください${NC}"
        echo -e "${YELLOW}MCPサーバーも一緒に停止されます${NC}"
        
        # 終了時にMCPサーバーも停止
        trap "echo -e '${YELLOW}サーバーを停止中...${NC}'; kill $MCP_PID 2>/dev/null; exit" INT TERM
        
        python "$APP_SERVER_FILE"
        ;;
    4)
        echo -e "${GREEN}テスト実行中...${NC}"
        cd "$PROJECT_ROOT"
        
        # MCPサーバーをバックグラウンドで起動
        nohup python "$MCP_SERVER_FILE" > mcp_server.log 2>&1 &
        MCP_PID=$!
        echo -e "${GREEN}MCPサーバー起動完了 (PID: $MCP_PID)${NC}"
        
        sleep 3
        
        # MCPサーバーのテスト
        echo -e "${YELLOW}MCPサーバーをテスト中...${NC}"
        curl -X POST "http://localhost:$MCP_PORT" \
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
             }'
        
        echo ""
        echo -e "${GREEN}テスト完了！${NC}"
        
        # MCPサーバーを停止
        kill $MCP_PID 2>/dev/null
        echo -e "${YELLOW}MCPサーバーを停止しました${NC}"
        ;;
    *)
        echo -e "${RED}無効な選択です${NC}"
        exit 1
        ;;
esac