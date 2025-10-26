# OpenWebUI + LangChain
OpenWebUIをインターフェイスとしたLLMサンプル

User --3000--> OpenWebUI --8000--> LangChain Python --11434--> LLM(Ollama)

## 実行方法
### OpenWebUi

```shell
openwebui$ cd openwebui-docker
openwebui-docker$ docker compose up -d
```

### LangChain server

```shell
openwebui$ venv/bin/python langchain_server/langchain_server.py
```
