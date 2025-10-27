import asyncio
import datetime
import json
import logging
import os
import sys
import time
import uuid
from fastapi import FastAPI, Request
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage
import uvicorn

# 設定ファイル名のデフォルト値
DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), 'app.json')

# Global variables to store initialized components
tools = None
llm = None
sys_message = None
config = None
begin_time = None
client = None
agent = None

app = FastAPI()

def setup_logging(log_folder: str) -> None:
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    filename = os.path.join(log_folder, f'mcp_ex_text.{datetime.datetime.now().strftime("%Y%m%d-%H%M")}.log')
    logging.basicConfig(
        filename=filename,
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )

def write_log(text: str, color: str = '0'):
    now = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    msg = f'{now} \033[{color}m{text}\033[0m'
    print(msg)
    logging.info(text)

def print_message_by_step(step, color_cfg):
    if 'tools' in step:
        for msg in step.get('tools').get('messages'):
            write_log(f'tool> {msg.content}', color=color_cfg['tool'])
    elif 'agent' in step:
        text = ''
        for msg in step['agent'].get('messages'):
            if msg.content != '' and not msg.tool_calls:
                text += msg.content
            for tool_call in msg.tool_calls:
                write_log(f'system> [{tool_call['name']}] {tool_call['args']}', color=color_cfg['system'])
        text = text.strip()
        if text != '':
            write_log(f'system> {text}', color=color_cfg['system'])
        return text
    return ''

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

async def initialize_components():
    global tools, llm, sys_message, config, begin_time, client, agent
    
    args = sys.argv
    config_path = args[1] if len(args) >= 2 else DEFAULT_CONFIG
    config = load_config(config_path)
    setup_logging(config['log_folder'])
    os.environ['TEMP'] = config['report_folder']
    begin_time = datetime.datetime.now()
    write_log(f'system> 開始します', color=config['color']['system'])

    # MultiServerMCPClientを初期化
    client = MultiServerMCPClient(config['mcp_servers'])
    tools = await client.get_tools()
    write_log(f'system> {len(tools)}個のツールをロードしました', color=config['color']['system'])
    for tool in tools:
        write_log(f'system> ツール: {tool.name} - {tool.description}', color=config['color']['system'])
    write_log(f'system> agentを構築します', color=config['color']['system'])
    sys_message = SystemMessage(content='\n'.join(config['system_prompts']))
    llm = ChatOpenAI(
        openai_api_base=config['openai_api_base'],
        streaming=False,
        temperature=0,
        openai_api_key="EMPTY",
        model=config['agent_model']
    )
    # agent = create_react_agent(llm, tools, prompt=sys_message, store=InMemoryStore())
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=sys_message,
        store=InMemoryStore()
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # 初期化されていない場合は初期化を実行
    if tools is None or llm is None or sys_message is None:
        await initialize_components()
    
    body = await request.json()
    messages = body.get("messages", [])
    print('input messages:', messages)
    # prompt = "\n".join([m["content"] for m in messages if m["role"] == "user"])
    prompt = "\n".join([m["content"] for m in messages])
    print('prompt', prompt)

    print('HumanMessage', HumanMessage(content=prompt))
    result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})

    write_log(f'agent> {result}', color=config['color']['agent'])
    end_time = datetime.datetime.now()
    write_log(f'system> 切断します (所要時間:{end_time - begin_time})', color=config['color']['system'])

    # LangChainの返答から最後のAIメッセージを抽出
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        ai_messages = [m for m in messages if getattr(m, "type", None) == "ai"]
        if ai_messages:
            response_text = ai_messages[-1].content
        else:
            response_text = str(result)
    else:
        # 通常の文字列やAIMessage型の場合
        response_text = getattr(result, "content", str(result))

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": config['agent_model'],
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(prompt.split()) + len(response_text.split())
        }
    }

@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {"id": "nautilus-llm", "object": "model"},
        ]
    }

if __name__ == '__main__':
    # アプリケーション起動時に初期化を実行
    import asyncio
    asyncio.run(initialize_components())
    uvicorn.run(app, host="0.0.0.0", port=8000)
