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

### Option 1: Local Installation

#### Steps

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
   cp .env.example .env
   # Edit .env with your API keys
   ```
   
   Required variables:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   SERPAPI_API_KEY=your_serpapi_api_key_here  # Optional

   # Optional: Disable LangSmith warnings
   LANGCHAIN_TRACING_V2=false
   ```

5. **Configure the pipeline**
   
   Edit `config.yaml` to customize:
   - ArXiv search query and number of papers
   - Chunking parameters
   - Model selections
   - Pinecone index settings

6. **Verify installation**
   ```bash
   python scripts/validate_setup.py
   ```

### Option 2: Docker Installation

#### Using Docker Compose (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/ashaduzzaman-sarker/ResearchAgent.git
   cd ResearchAgent
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   Open your browser to `http://localhost:8501`

5. **View logs**
   ```bash
   docker-compose logs -f
   ```

6. **Stop the application**
   ```bash
   docker-compose down
   ```

#### Using Docker directly

```bash
# Build the image
docker build -t research-agent .

# Run the container
docker run -p 8501:8501 --env-file .env research-agent
```

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
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Run Individual Components

You can run each component of the pipeline independently:

```bash
# Only extract data from ArXiv
python src/data_extraction.py

# Only process and chunk PDFs
python src/data_processing.py

# Only generate embeddings
python src/embeddings.py

# Only run the agent (requires prior steps to be completed)
python src/agent_graph.py
```

### Development & Testing

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/
```

## 📁 Project Structure

```
ResearchAgent/
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD pipeline configuration
├── src/
│   ├── __init__.py
│   ├── main.py                    # Pipeline orchestrator
│   ├── data_extraction.py         # ArXiv data extraction
│   ├── data_processing.py         # PDF processing & chunking
│   ├── embeddings.py              # Embedding generation & indexing
│   ├── tools.py                   # RAG & web search tools
│   └── agent_graph.py             # LangGraph workflow
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Test fixtures and configuration
│   ├── test_data_extraction.py   # Unit tests for data extraction
│   ├── test_data_processing.py   # Unit tests for data processing
│   └── test_tools.py              # Unit tests for RAG and web search
├── notebooks/
│   └── research_agent.ipynb       # Jupyter notebook demo
├── streamlit_app.py               # Streamlit web interface
├── config.yaml                    # Pipeline configuration
├── pyproject.toml                 # Project metadata and build config
├── requirements.txt               # Production dependencies
├── requirements-dev.txt           # Development dependencies
├── .env.example                   # Environment variables template
├── CONTRIBUTING.md                # Contribution guidelines
├── files/                         # Generated data files (created at runtime)
│   ├── arxiv_dataset.json
│   ├── expanded_dataset.json
│   ├── embedding_metadata.json
│   ├── agent_responses.json
│   └── pdfs/                      # Downloaded PDF files
└── logs/                          # Application logs (created at runtime)
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

- **[LangChain](https://github.com/langchain-ai/langchain)** (v1.2.0): Framework for LLM applications
- **[LangGraph](https://github.com/langchain-ai/langgraph)** (v1.0.5): Graph-based agent orchestration
- **[OpenAI API](https://openai.com/)** (v2.13.0): GPT-4o for reasoning, text-embedding-ada-002 for embeddings
- **[Pinecone](https://www.pinecone.io/)** (v8.0.0): Serverless vector database for semantic search
- **[Streamlit](https://streamlit.io/)** (v1.52.2): Interactive web interface
- **[SerpAPI](https://serpapi.com/)** (v2.4.2): Web search integration
- **[ArXiv API](https://arxiv.org/help/api)** (v2.3.1): Research paper access
- **[PyPDF](https://pypdf.readthedocs.io/)** (v6.4.2): PDF processing

## 🧪 Testing & Quality Assurance

ResearchAgent includes comprehensive testing and quality checks:

- **Unit Tests**: Core functionality testing for all modules
- **Integration Tests**: End-to-end pipeline validation
- **Code Coverage**: Target >80% coverage
- **Linting**: Ruff for code quality
- **Formatting**: Black for consistent code style
- **Type Checking**: MyPy for type safety
- **Security Scanning**: Bandit for security vulnerabilities
- **CI/CD**: GitHub Actions for automated testing

Run the full test suite:
```bash
pytest --cov=src --cov-report=term-missing
```

## 🚀 CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

- **Automated Testing**: Runs on every push and pull request
- **Code Quality Checks**: Linting, formatting, and type checking
- **Security Scans**: Vulnerability detection with Bandit and Safety
- **Package Building**: Validates package distribution
- **Coverage Reports**: Tracks test coverage over time

## 📈 Performance Considerations

- **Query Response Time**: 5-15 seconds (varies by tool usage)
- **Embedding Generation**: ~2-3 papers/minute (rate limited by OpenAI API)
- **Vector Search**: <1 second for top-5 retrieval from Pinecone
- **Scalability**: Supports 1000+ papers in knowledge base
- **Chunk Size**: 800 characters with 100-character overlap (configurable)
- **Batch Processing**: Embeddings generated in batches of 32

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

- Setting up your development environment
- Code style and standards
- Testing requirements
- Submitting pull requests
- Reporting issues

For major changes, please open an issue first to discuss your ideas.

## 🔧 Troubleshooting

### Common Issues

**1. API Key Errors**
```
Error: OPENAI_API_KEY not found
```
**Solution**: Ensure your `.env` file exists and contains valid API keys. Copy `.env.example` to `.env` and fill in your keys.

**2. Pinecone Index Not Found**
```
Error: Failed to connect to Pinecone index
```
**Solution**: The index is created automatically on first run. Ensure your `PINECONE_API_KEY` is valid and you have proper permissions.

**3. PDF Download Failures**
```
Failed to download PDF: Connection timeout
```
**Solution**: ArXiv may rate-limit requests. The system will skip failed downloads and continue. Rerun the pipeline to retry.

**4. Streamlit Port Already in Use**
```
Error: Address already in use
```
**Solution**: Change the port in `config.yaml` or use: `streamlit run streamlit_app.py --server.port 8502`

**5. Import Errors**
```
ModuleNotFoundError: No module named 'langchain'
```
**Solution**: Ensure you've activated your virtual environment and installed dependencies:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Performance Optimization

- **Reduce max_results** in `config.yaml` if downloads are slow
- **Adjust chunk_size** to balance between context and precision
- **Use batch_size** efficiently for embedding generation
- **Enable LangSmith** for debugging and monitoring (optional)

## 📚 API Reference

### Configuration Options (config.yaml)

```yaml
data_extraction:
  arxiv:
    search_query: "cat:cs.AI"        # ArXiv search query
    max_results: 100                 # Number of papers to fetch
  output:
    json_file_path: "files/arxiv_dataset.json"
    pdf_folder: "files/pdfs"

data_processing:
  chunking:
    chunk_size: 800                  # Characters per chunk
    chunk_overlap: 100               # Overlap between chunks
  output:
    expanded_json_path: "files/expanded_dataset.json"

embeddings:
  model: "text-embedding-ada-002"    # OpenAI embedding model
  batch_size: 32                     # Batch size for embedding generation
  pinecone:
    index_name: "research-agent-index"
    namespace: "arxiv_chunks"
    dimension: 1536                  # Embedding dimension
  output:
    embedding_metadata_path: "files/embedding_metadata.json"

agent:
  llm_model: "gpt-4o"                # LLM for answer generation
  max_iterations: 5                  # Max agent iterations
  tools:
    rag:
      top_k: 5                       # Number of documents to retrieve
    serpapi:
      max_results: 3                 # Number of web results
  output:
    responses_path: "files/agent_responses.json"
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for embeddings and LLM |
| `PINECONE_API_KEY` | Yes | Pinecone API key for vector database |
| `SERPAPI_API_KEY` | Optional | SerpAPI key for web search (agent will skip web search if not provided) |
| `LANGCHAIN_TRACING_V2` | Optional | Enable LangSmith tracing (true/false) |
| `LANGCHAIN_API_KEY` | Optional | LangSmith API key for monitoring |
| `LANGCHAIN_PROJECT` | Optional | LangSmith project name |

### Module API

#### Data Extraction
```python
from src.data_extraction import extract_arxiv_data, download_pdfs

# Extract paper metadata
df = extract_arxiv_data(
    search_query="cat:cs.AI",
    max_results=10,
    json_file_path="output.json"
)

# Download PDFs
df = download_pdfs(df, download_folder="pdfs")
```

#### Data Processing
```python
from src.data_processing import load_and_chunk_pdf, expand_df

# Chunk a single PDF
chunks = load_and_chunk_pdf(
    "paper.pdf",
    chunk_size=800,
    chunk_overlap=100
)

# Expand DataFrame with chunks
expanded_df = expand_df(df, chunk_size=800, chunk_overlap=100)
```

#### Embeddings
```python
from src.embeddings import generate_and_index_embeddings, initialize_pinecone

# Initialize Pinecone index
index = initialize_pinecone(
    index_name="my-index",
    dimension=1536
)

# Generate and index embeddings
generate_and_index_embeddings(
    df,
    model_name="text-embedding-ada-002",
    index_name="my-index",
    namespace="chunks"
)
```

#### Agent Tools
```python
from src.tools import rag_search, web_search

# RAG search
results = rag_search.invoke({
    "query": "What is attention mechanism?",
    "index_name": "research-agent-index",
    "namespace": "arxiv_chunks",
    "top_k": 5
})

# Web search
results = web_search.invoke({
    "query": "latest AI news",
    "max_results": 3
})
```

#### Agent Graph
```python
from src.agent_graph import build_graph

# Build and run agent
graph = build_graph()
result = graph.invoke({
    "query": "Explain transformers",
    "messages": [],
    "retrieved_docs": "",
    "web_results": "",
    "final_answer": "",
    "tools_to_use": [],
    "iteration": 0
})

print(result["final_answer"])
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

