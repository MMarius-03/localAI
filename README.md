# LocalAI

A local FastAPI service that connects Ollama models to Python tools for documents, charts, web search, and email. The service runs on your machine; web search and email use external services when configured.

## Run locally

Install Python dependencies and start Ollama before running the service. By default, it expects `qwen3:4b-instruct` for tool calls and `qwen3:14b` for chat. You can change these with `TOOL_MODEL` and `CHAT_MODEL`.

```bash
git clone https://github.com/MMarius-03/localAI.git
cd localAI
python -m venv .venv
pip install -r requirements.txt
python code.py
```

The API runs at `http://localhost:8000`. SearXNG is optional for web search. Email requires SMTP settings.

## What it does

- Creates and edits PDF, DOCX, and HTML documents.
- Generates charts with Matplotlib.
- Searches through SearXNG and extracts page content.
- Sends email when SMTP credentials are configured.

The implementation is in [`code.py`](code.py).
