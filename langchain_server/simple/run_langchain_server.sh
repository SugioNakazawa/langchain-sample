#!/bin/bash

# LangChain Server 実行スクリプト
# 使用方法: ./run_langchain_server.sh

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
SERVER_FILE="$SCRIPT_DIR/langchain_server.py"
SERVER_HOST="0.0.0.0"
SERVER_PORT="8000"

echo -e "${BLUE}=== LangChain Server 起動スクリプト ===${NC}"
echo -e "${YELLOW}プロジェクトディレクトリ: $PROJECT_ROOT${NC}"
echo -e "${YELLOW}仮想環境: $VENV_PATH${NC}"
echo -e "${YELLOW}サーバーファイル: $SERVER_FILE${NC}"

# 仮想環境の確認
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}エラー: 仮想環境が見つかりません: $VENV_PATH${NC}"
    echo -e "${YELLOW}以下のコマンドで仮想環境を作成してください:${NC}"
    echo "  cd $PROJECT_ROOT"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install fastapi uvicorn langchain langchain-openai"
    exit 1
fi

# サーバーファイルの確認
if [ ! -f "$SERVER_FILE" ]; then
    echo -e "${RED}エラー: サーバーファイルが見つかりません: $SERVER_FILE${NC}"
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
if lsof -i :$SERVER_PORT > /dev/null 2>&1; then
    echo -e "${YELLOW}警告: ポート $SERVER_PORT は既に使用されています${NC}"
    echo -e "${YELLOW}続行しますか? (y/N)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# サーバー起動
echo -e "${GREEN}LangChain Serverを起動しています...${NC}"
echo -e "${BLUE}URL: http://$SERVER_HOST:$SERVER_PORT${NC}"
echo -e "${BLUE}API: http://$SERVER_HOST:$SERVER_PORT/v1/chat/completions${NC}"
echo -e "${BLUE}Models: http://$SERVER_HOST:$SERVER_PORT/v1/models${NC}"
echo -e "${YELLOW}停止するには Ctrl+C を押してください${NC}"
echo ""

cd "$PROJECT_ROOT"
python "$SERVER_FILE"