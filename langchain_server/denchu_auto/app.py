import asyncio
import datetime
import json
import logging
import os
import sys
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langgraph.store.memory import InMemoryStore

# 設定ファイル名のデフォルト値
DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), 'app.json')

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

async def main():
    args = sys.argv
    config_path = args[1] if len(args) >= 2 else DEFAULT_CONFIG
    config = load_config(config_path)
    setup_logging(config['log_folder'])
    os.environ['TEMP'] = config['report_folder']
    begin_time = datetime.datetime.now()
    write_log(f'system> 開始します', color=config['color']['system'])
    
    # MultiServerMCPClientを直接初期化（async withを使用しない）
    client = MultiServerMCPClient(config['mcp_servers'])
    try:
        tools = await client.get_tools()
        write_log(f'system> {len(tools)}個のツールをロードしました', color=config['color']['system'])
        for tool in tools:
            write_log(f'system> ツール: {tool.name} - {tool.description}', color=config['color']['system'])
        
        llm = ChatOpenAI(
            openai_api_base=config['openai_api_base'],
            streaming=False,
            temperature=0,
            openai_api_key="EMPTY",
            model=config['agent_model']
        )
        
        write_log(f'system> agentを構築します', color=config['color']['system'])
        sys_message = SystemMessage(content='\n'.join(config['system_prompts']))
        agent = create_react_agent(llm, tools, prompt=sys_message, store=InMemoryStore())
        
        messages = []
        for input_text in config['input_texts']:
            write_log(f'user> {input_text}', color=config['color']['user'])
            last_msg = None
            message = {'role': 'user', 'content': input_text}
            messages.append(message)
            async for step in agent.astream({'messages': messages}):
                last_msg = print_message_by_step(step, config['color'])
            if last_msg is not None:
                messages.append({'role': 'assistant', 'content': last_msg})
        
        if config.get('show_report'):
            report_folder = config['report_folder']
            if not os.path.exists(report_folder):
                os.makedirs(report_folder)
            files = os.listdir(report_folder)
            report_files = [f for f in files if f.startswith('checklist-result-') and f.endswith('.xlsx')]
            if report_files:
                recent_file = max(report_files, key=lambda f: os.path.getmtime(os.path.join(report_folder, f)))
                write_log(f'system> レポートを開きます...{recent_file}', color=config['color']['system'])
                if sys.platform == "win32":
                    os.system(f'powershell -Command "{report_folder}\\{recent_file}"')
                elif sys.platform == "darwin":
                    os.system(f'open "{os.path.join(report_folder, recent_file)}"')
                else:
                    os.system(f'xdg-open "{os.path.join(report_folder, recent_file)}"')
        
        end_time = datetime.datetime.now()
        write_log(f'system> 切断します (所要時間:{end_time - begin_time})', color=config['color']['system'])
    
    finally:
        # クライアントのクリーンアップ（必要に応じて）
        if hasattr(client, 'close'):
            try:
                await client.close()
            except:
                pass

if __name__ == '__main__':
    asyncio.run(main())