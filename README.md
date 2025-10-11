# Research Agent

A LangGraph-based agent for research tasks using GPT-4o, RAG (Pinecone), ArXiv, and SerpAPI.

## Setup

1. Clone the repo: `git clone https://github.com/ashaduzzaman-sarker/ResearchAgent.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your API keys (OPENAI_API_KEY, PINECONE_API_KEY, SERPAPI_API_KEY).
4. Run: `python src/main.py`

## Project Structure

ResearchAgent/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── config.yaml
├── logs
├── src/
│   ├── __init__.py
│   ├── data_extraction.py  # For ArXiv extraction and PDF handling
│   ├── data_processing.py  # For chunking and expanding DataFrame
│   ├── embeddings.py       # For embedding and Pinecone indexing (added)
│   ├── tools.py            # For custom tools like RAG search, web search
│   ├── agent_graph.py     # For LangGraph workflow
│   └── main.py            # Entry point to run the agent
├── files/                 # For storing JSON and PDFs (gitignored)
│   ├── arxiv_dataset.json
│   └── pdfs/
└── tests/                 # Optional: For unit tests (added for robustness)
    └── test_data_extraction.py

## Usage

Query the agent: `agent.invoke({"input": "Your query here", "chat_history": []})`

## Customization

- Swap data sources: Edit `data_extraction.py` (e.g., replace ArXiv with PubMed).
- Change LLM: Update `agent_graph.py` to use another model.
- Add tools: Extend `tools.py`.

## License

MIT