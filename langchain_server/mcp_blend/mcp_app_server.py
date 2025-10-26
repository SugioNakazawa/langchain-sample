from fastapi import FastAPI, Request
from langchain_mcp_agent import get_optimal_blend
import uvicorn

app = FastAPI()

@app.post("/v1/chat/completions")
async def completions(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    if any("最適ブレンド" in m["content"] for m in messages):
        reply = get_optimal_blend()
    else:
        reply = "最適化コマンドを認識できません。"
    return {"choices": [{"message": {"role": "assistant", "content": reply}}]}

@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {"id": "nautilus-llm", "object": "model"},
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
