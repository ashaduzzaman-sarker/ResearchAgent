import os
import sys
import json
import logging
import yaml
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.agent_graph import build_graph

# ================================
# Logging Configuration
# ================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/research_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================
# Utility Functions
# ================================
@st.cache_resource
def load_config(config_path="config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded from %s", config_path)
        return config
    except FileNotFoundError:
        logger.error("Configuration file %s not found", config_path)
        st.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error("Error parsing configuration file: %s", e)
        st.error(f"Error parsing configuration file: {e}")
        raise

def load_env():
    """Load environment variables."""
    try:
        load_dotenv()
        logger.info("Environment variables loaded")
    except Exception as e:
        logger.error("Error loading environment variables: %s", e)
        st.warning("Failed to load .env file")

@st.cache_resource
def initialize_graph():
    """Initialize the LangGraph agent."""
    try:
        graph = build_graph()
        logger.info("LangGraph initialized successfully")
        return graph
    except Exception as e:
        logger.error("Failed to initialize graph: %s", e)
        st.error("Failed to initialize agent graph.")
        raise

def run_agent(query: str, graph, config: dict) -> dict:
    """Execute agent query and return structured response."""
    if not query or not isinstance(query, str) or not query.strip():
        logger.warning("Empty or invalid query submitted")
        return {"error": "Invalid or empty query"}

    try:
        initial_state = {
            "query": query,
            "messages": [{"role": "user", "content": query}],
            "retrieved_docs": "",
            "web_results": "",
            "final_answer": "",
            "tools_to_use": [],
            "iteration": 0
        }

        logger.info("Executing agent for query: %s", query)
        result = graph.invoke(initial_state)

        response_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "tools_used": result.get("tools_to_use", []),
            "retrieved_docs": result.get("retrieved_docs", ""),
            "web_results": result.get("web_results", ""),
            "final_answer": result.get("final_answer", "No answer generated"),
            "messages": result.get("messages", [])
        }

        # Save response to file
        output_json = config.get("agent", {}).get("output", {}).get("responses_path", "outputs/responses.json")
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        existing_data = []
        if os.path.exists(output_json):
            with open(output_json, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        existing_data.append(response_data)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
        logger.info("Saved response to %s", output_json)

        return response_data

    except Exception as e:
        logger.error("Agent execution error for query '%s': %s", query, e)
        return {"error": f"Agent execution failed: {e}"}

# ================================
# Streamlit UI
# ================================
def main():
    st.set_page_config(page_title="Research Agent Interface", page_icon="🔍", layout="wide")

    load_env()
    config = load_config()
    graph = initialize_graph()

    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;  /* light gray background */
        color: #000000;              /* black text */
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        overflow-wrap: break-word;
    }
    </style>
""", unsafe_allow_html=True)


    st.markdown('<div class="main-header">🔍 Research Agent Interface</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered research assistant for ArXiv papers and web search</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This Research Agent can:
        - 🔬 Search ArXiv AI/ML papers
        - 🌐 Perform web searches
        - 🤖 Generate GPT-4 powered answers
        """)

        st.header("⚙️ Configuration")
        st.markdown(f"""
        - **LLM Model**: {config['agent'].get('llm_model', 'Not set')}
        - **RAG Top-K**: {config['agent']['tools']['rag'].get('top_k', 'N/A')}
        - **Web Results**: {config['agent']['tools']['serpapi'].get('max_results', 'N/A')}
        - **Index**: {config['embeddings']['pinecone'].get('index_name', 'N/A')}
        """)

        st.header("📝 Example Queries")
        example_queries = [
            "Latest advancements in reinforcement learning?",
            "Explain transformer architecture in deep learning",
            "Compare supervised vs unsupervised learning",
            "What is few-shot learning?",
            "Recent developments in computer vision"
        ]
        for eq in example_queries:
            if st.button(eq, key=eq):
                st.session_state.query_input = eq

    # Query input and buttons
    query = st.text_input(
        "Enter your research query:",
        placeholder="e.g., Latest advancements in RL",
        key="query_input",
        value=st.session_state.get("query_input", "")
    )

    col1, col2 = st.columns([1, 1])
    search_button = col1.button("🔍 Search", type="primary")
    clear_button = col2.button("🗑️ Clear")

    if clear_button:
        st.session_state.query_input = ""
        st.rerun()

    if search_button:
        if not query.strip():
            st.error("⚠️ Please enter a valid query.")
            return

        # Check for API keys before making requests
        required_keys = {
            "OPENAI_API_KEY": "OpenAI API key is required for embeddings and LLM",
            "PINECONE_API_KEY": "Pinecone API key is required for vector search"
        }
        
        missing_keys = []
        for key, description in required_keys.items():
            if not os.getenv(key):
                missing_keys.append(f"❌ {description}")
        
        if missing_keys:
            st.error("**Missing Required API Keys:**")
            for msg in missing_keys:
                st.error(msg)
            st.info("💡 Please configure your .env file with the required API keys. See .env.example for reference.")
            return

        with st.spinner("🤔 Processing your query..."):
            try:
                response = run_agent(query, graph, config)
            except Exception as e:
                st.error(f"❌ An error occurred while processing your query: {str(e)}")
                logger.error(f"Streamlit error for query '{query}': {e}", exc_info=True)
                return

        if "error" in response:
            st.error(f"❌ Error: {response['error']}")
            st.info("💡 Try rephrasing your query or check the logs for more details.")
            return

        # Display tabs for results
        tabs = st.tabs(["📝 Answer", "📚 ArXiv Papers", "🌐 Web Results", "💬 Message History"])
        with tabs[0]:
            st.markdown("### 📝 Final Answer")
            st.markdown(f'<div class="result-box">{response.get("final_answer", "No answer generated")}</div>', unsafe_allow_html=True)
            tools_used = response.get("tools_used", [])
            if tools_used:
                st.markdown(f"**Tools Used**: {', '.join(tools_used)}")

        with tabs[1]:
            st.markdown("### 📚 Retrieved ArXiv Documents")
            st.text_area("Documents", response.get("retrieved_docs", "No ArXiv documents retrieved."), height=400)

        with tabs[2]:
            st.markdown("### 🌐 Web Search Results")
            st.text_area("Web Results", response.get("web_results", "No web results retrieved."), height=400)

        with tabs[3]:
            st.markdown("### 💬 Message History")
            messages = response.get("messages", [])
            if messages:
                for msg in messages:
                    role = msg.get("role", "unknown").capitalize()
                    content = msg.get("content", "")
                    color = "blue" if role == "User" else "green" if role == "Assistant" else "black"
                    st.markdown(f"**:<span style='color:{color}'>{role}</span>**: {content}", unsafe_allow_html=True)
                    st.divider()
            else:
                st.info("No message history available.")

        # Download JSON
        st.download_button(
            "📥 Download Response (JSON)",
            data=json.dumps(response, indent=2, ensure_ascii=False),
            file_name=f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    # Footer
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#666;font-size:0.9rem;'>Built with ❤️ using LangChain, LangGraph, Pinecone, and Streamlit</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
