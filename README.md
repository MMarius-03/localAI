# LocalAI Agent Service

An agentic backend powered by **FastAPI** and **Ollama** that transforms LLM prompts into actionable tasks: generating PDFs, creating charts, scraping websites, and sending emails—all running locally.

##  Features
- **Native Tool Calling:** Uses `qwen3` (or similar) to execute Python-based tools.
- **Document Suite:** Create/Merge/Split PDFs, DOCX, and HTML.
- **Data Viz:** Generate Bar, Line, and Pie charts via Matplotlib.
- **Web Intelligence:** Integrated SearXNG search and BeautifulSoup scraping.
- **Communication:** Send emails with attachments directly from the agent.

##  Quick Start

1. **Clone & Install:**
   ```bash
   git clone [https://github.com/youruser/localAI-agent.git](https://github.com/youruser/localAI-agent.git)
   pip install -r requirements.txt
