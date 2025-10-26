from fastapi import FastAPI, Request
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import uvicorn

# Simpleserver to expose a local LLM via OpenAI-compatible API

app = FastAPI()

# 例: ローカルLLM (Ollamaなど) に接続
llm = ChatOpenAI(
    openai_api_base="http://localhost:11434/v1",  # Ollama
    openai_api_key="none",
    model="qwen3:4b"
)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    prompt = "\n".join([m["content"] for m in messages if m["role"] == "user"])

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": response.content}}],
    }

@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {"id": "nautilus-llm", "object": "model"},
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
