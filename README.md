# 🔬 Research Agent

> An intelligent AI-powered research assistant that searches through ArXiv papers and the web to provide comprehensive answers to your questions.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-green.svg)](https://github.com/langchain-ai/langchain)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

## 🎯 Overview

Research Agent is an end-to-end AI research assistant that combines **Retrieval-Augmented Generation (RAG)** with **web search** capabilities to answer complex research questions. It automatically extracts papers from ArXiv, processes them into searchable chunks, stores embeddings in Pinecone, and uses GPT-4 to generate intelligent responses.

## ✨ Features

- 🔍 **Intelligent Query Classification**: Automatically determines whether to use RAG search, web search, or both
- 📚 **ArXiv Integration**: Fetches and processes the latest AI/ML research papers
- 🧠 **Vector Search**: Uses Pinecone for efficient semantic search over research papers
- 🌐 **Web Search**: Integrates SerpAPI for current information and general knowledge
- 🤖 **GPT-4 Powered**: Generates comprehensive, well-cited answers
- 💬 **Interactive UI**: Beautiful Streamlit interface for easy interaction
- 🔄 **Complete Pipeline**: Automated workflow from data extraction to answer generation
- 📊 **Logging & Monitoring**: Comprehensive logging for debugging and monitoring

## 🏗️ Architecture

```
┌─────────────┐
│  User Query │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Query Classifier │ ◄─── GPT-4o-mini
└──────┬───────────┘
       │
       ├─────────────┬─────────────┐
       ▼             ▼             ▼
  ┌────────┐   ┌──────────┐   ┌─────────┐
  │  RAG   │   │   Web    │   │  Both   │
  │ Search │   │  Search  │   │ Enabled │
  └────┬───┘   └────┬─────┘   └────┬────┘
       │            │              │
       └────────────┴──────────────┘
                    │
                    ▼
          ┌──────────────────┐
          │ Answer Generator │ ◄─── GPT-4o
          └────────┬─────────┘
                   │
                   ▼
            ┌─────────────┐
            │ Final Answer│
            └─────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.12 or higher
- OpenAI API key
- Pinecone API key
- SerpAPI key (optional, for web search)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/ashaduzzaman-sarker/ResearchAgent.git
   cd ResearchAgent
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   SERPAPI_API_KEY=your_serpapi_api_key_here

   # Optional: Disable LangSmith warnings
   LANGCHAIN_TRACING_V2=false
   ```

5. **Configure the pipeline**
   
   Edit `config.yaml` to customize:
   - ArXiv search query and number of papers
   - Chunking parameters
   - Model selections
   - Pinecone index settings

## ⚙️ Configuration

The `config.yaml` file controls all aspects of the pipeline:

```yaml
data_extraction:
  arxiv:
    search_query: "cat:cs.AI"  # ArXiv category
    max_results: 100             # Number of papers to fetch

data_processing:
  chunking:
    chunk_size: 800             # Characters per chunk
    chunk_overlap: 100          # Overlap between chunks

embeddings:
  model: "text-embedding-ada-002"
  pinecone:
    index_name: "research-agent-index"
    namespace: "arxiv_chunks"

agent:
  llm_model: "gpt-4o"          # Main LLM for answers
  tools:
    rag:
      top_k: 5                  # Number of documents to retrieve
    serpapi:
      max_results: 3            # Number of web results
```

## 💻 Usage

### Run the Complete Pipeline

Execute the full pipeline (extraction → processing → embedding → agent):

```bash
python src/main.py
```

This will:
1. Extract papers from ArXiv
2. Download and process PDFs
3. Generate embeddings and index to Pinecone
4. Run a test query through the agent

### Launch the Web Interface

Start the Streamlit app for an interactive experience:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Run Individual Components

```bash
# Only extract data from ArXiv
python src/data_extraction.py

# Only process and chunk PDFs
python src/data_processing.py

# Only generate embeddings
python src/embeddings.py

# Only run the agent
python src/agent_graph.py
```

## 📁 Project Structure

```
ResearchAgent/
├── src/
│   ├── __init__.py
│   ├── main.py                    # Pipeline orchestrator
│   ├── data_extraction.py         # ArXiv data extraction
│   ├── data_processing.py         # PDF processing & chunking
│   ├── embedding.py          # Embedding generation
│   ├── tools.py                   # RAG & web search tools
│   └── agent_graph.py             # LangGraph workflow
├── streamlit_app.py                         # Streamlit interface
├── config.yaml                    # Configuration file
├── .env.example                   # Environment variables (create .env)
├── requirements.txt               # Python dependencies
├── files/                         # Create 'files' for Generated data files
│   ├── arxiv_dataset.json
│   ├── expanded_dataset.json
│   ├── embedding_metadata.json
│   ├── agent_responses.json
│   └── pdfs/                      # Downloaded PDFs
└── logs/                          # Create 'logs' for Log files
    ├── main_pipeline.log
    └── research_agent.log
```

## 🔍 How It Works

### 1. Data Extraction
- Searches ArXiv using specified query (e.g., "cat:cs.AI" for AI papers)
- Downloads paper metadata and PDFs
- Stores information in JSON format

### 2. Data Processing
- Loads PDFs using LangChain's PyPDFLoader
- Splits documents into overlapping chunks for better context
- Creates expanded dataset with chunk metadata

### 3. Embedding Generation
- Generates embeddings using OpenAI's text-embedding-ada-002
- Indexes embeddings in Pinecone vector database
- Stores metadata for retrieval

### 4. Agent Execution
- **Query Classification**: LLM determines which tools to use
- **RAG Search**: Retrieves relevant chunks from Pinecone
- **Web Search**: Fetches current information from the web (optional)
- **Answer Generation**: GPT-4 synthesizes information into a comprehensive answer

### 5. LangGraph Workflow

The agent uses a state machine workflow:

```
START → Classify Query → [RAG Search] → [Web Search] → Generate Answer → END
                     ↓                                        ↑
                     └────────────────────────────────────────┘
```

## 📊 Performance

- **Query Response Time**: 5-15 seconds (depending on tools used)
- **Embedding Generation**: ~2-3 papers/minute
- **Vector Search**: <1 second for top-5 retrieval
- **Supports**: Up to 1000+ papers in the knowledge base


## 📝 Example Queries

Try these queries in the interface:

- "What are the latest advancements in reinforcement learning?"
- "Explain transformer architecture in deep learning"
- "Compare supervised and unsupervised learning approaches"
- "What is few-shot learning and how does it work?"
- "Recent developments in computer vision"

## 🛠️ Technologies Used

- **[LangChain](https://github.com/langchain-ai/langchain)**: Framework for LLM applications
- **[LangGraph](https://github.com/langchain-ai/langgraph)**: Graph-based agent orchestration
- **[OpenAI API](https://openai.com/)**: GPT-4o for reasoning, text-embedding-ada-002 for embeddings
- **[Pinecone](https://www.pinecone.io/)**: Vector database for semantic search
- **[Streamlit](https://streamlit.io/)**: Interactive web interface
- **[SerpAPI](https://serpapi.com/)**: Web search integration
- **[ArXiv API](https://arxiv.org/help/api)**: Research paper access
- **[PyPDF](https://pypdf.readthedocs.io/)**: PDF processing


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

