import requests
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# MCPクライアント設定（シンプルなHTTPクライアント）
class SimpleMCPClient:
    def __init__(self, url):
        self.url = url
    
    def invoke_tool(self, tool_id, input_data):
        payload = {
            "tool_id": tool_id,
            "input": input_data
        }
        response = requests.post(self.url, json=payload)
        return response.json()

client = SimpleMCPClient("http://localhost:9100")

# 使用LLM（ローカルLLMやOllamaでもOK）
llm = ChatOpenAI(
    openai_api_base="http://localhost:11434/v1",  # Ollama例
    openai_api_key="none",
    model="qwen3:14b"
)

def get_optimal_blend():
    oils = [
        {"name": "Soybean", "cost": 1.1, "iodine": 120},
        {"name": "Palm", "cost": 0.9, "iodine": 80},
        {"name": "Rapeseed", "cost": 1.0, "iodine": 110},
    ]

    result = client.invoke_tool("optimize_blend", {"oils": oils, "demand": 1000})
    blend = result["output"]["blend"]
    total = result["output"]["total_cost"]

    prompt = f"""
    以下の最適化結果をもとに、来週のブレンド方針を技術者向けに説明してください。
    結果: {blend}, 総コスト: {total:.2f}
    """
    msg = HumanMessage(content=prompt)
    response = llm.invoke([msg])
    return response.content

if __name__ == "__main__":
    print(get_optimal_blend())
