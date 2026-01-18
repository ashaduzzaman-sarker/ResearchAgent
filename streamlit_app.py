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

def run_agent(query: str, graph, config: dict, resume_state: dict | None = None, approved: bool = False) -> dict:
    """Execute agent query and return structured response."""
    if not query or not isinstance(query, str) or not query.strip():
        logger.warning("Empty or invalid query submitted")
        return {"error": "Invalid or empty query"}

    try:
        if resume_state:
            initial_state = resume_state
            initial_state["approved"] = approved
            initial_state["awaiting_approval"] = False
            initial_state["stage"] = "approved"
        else:
            initial_state = {
                "query": query,
                "messages": [{"role": "user", "content": query}],
                "retrieved_docs": "",
                "web_results": "",
                "tools_to_use": [],
                "iteration": 0,
                "research_notes": "",
                "research_summary": "",
                "sources": "",
                "research_complete": False,
                "approval_required": config.get("agent", {}).get("hitl", {}).get("enabled", True),
                "approved": not config.get("agent", {}).get("hitl", {}).get("enabled", True),
                "awaiting_approval": False,
                "draft_report": "",
                "edited_report": "",
                "fact_check_report": "",
                "final_report": "",
                "final_answer": "",
                "stage": "initialized"
            }

        logger.info("Executing agent for query: %s", query)
        result = graph.invoke(initial_state)

        response_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "tools_used": result.get("tools_to_use", []),
            "retrieved_docs": result.get("retrieved_docs", ""),
            "web_results": result.get("web_results", ""),
            "research_notes": result.get("research_notes", ""),
            "draft_report": result.get("draft_report", ""),
            "edited_report": result.get("edited_report", ""),
            "fact_check_report": result.get("fact_check_report", ""),
            "final_report": result.get("final_report", ""),
            "final_answer": result.get("final_answer", "No answer generated"),
            "awaiting_approval": result.get("awaiting_approval", False),
            "stage": result.get("stage", ""),
            "messages": result.get("messages", []),
            "state": result
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

        # Save final reports separately when available
        if response_data.get("final_report"):
            reports_path = config.get("agent", {}).get("output", {}).get("reports_path", "outputs/reports.json")
            os.makedirs(os.path.dirname(reports_path), exist_ok=True)
            reports_data = []
            if os.path.exists(reports_path):
                with open(reports_path, "r", encoding="utf-8") as f:
                    try:
                        reports_data = json.load(f)
                    except json.JSONDecodeError:
                        reports_data = []
            reports_data.append({
                "timestamp": response_data["timestamp"],
                "query": query,
                "final_report": response_data["final_report"],
                "fact_check_report": response_data.get("fact_check_report", "")
            })
            with open(reports_path, "w", encoding="utf-8") as f:
                json.dump(reports_data, f, ensure_ascii=False, indent=4)
            logger.info("Saved report to %s", reports_path)

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


    st.markdown('<div class="main-header">🧭 Multi-Agent Research Team</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Researcher → Writer → Editor → Fact-Checker with Human-in-the-loop approval</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This Multi-Agent Team can:
        - 🔬 Research from ArXiv and the web
        - ✍️ Draft a deep-dive report (~2,000 words)
        - ✏️ Edit for clarity and structure
        - ✅ Fact-check and add caveats
        - 🛑 Pause for human approval before writing
        """)

        st.header("⚙️ Configuration")
        st.markdown(f"""
        - **LLM Model**: {config['agent'].get('llm_model', 'Not set')}
        - **RAG Top-K**: {config['agent']['tools']['rag'].get('top_k', 'N/A')}
        - **Web Provider**: {config['agent']['tools']['web'].get('provider', 'N/A')}
        - **Web Results**: {config['agent']['tools']['web'].get('max_results', 'N/A')}
        - **Index**: {config['embeddings']['pinecone'].get('index_name', 'N/A')}
        - **HITL Enabled**: {config['agent']['hitl'].get('enabled', True)}
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
        web_provider = config["agent"]["tools"]["web"].get("provider", "serper").lower()
        required_keys = {
            "OPENAI_API_KEY": "OpenAI API key is required for embeddings and LLM",
            "PINECONE_API_KEY": "Pinecone API key is required for vector search"
        }
        if web_provider == "serper":
            required_keys["SERPER_API_KEY"] = "Serper API key is required for web search"
        else:
            required_keys["SERPAPI_API_KEY"] = "SerpAPI key is required for web search"
        
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
                st.session_state.last_response = response
            except Exception as e:
                st.error(f"❌ An error occurred while processing your query: {str(e)}")
                logger.error(f"Streamlit error for query '{query}': {e}", exc_info=True)
                return

    response = st.session_state.get("last_response")
    if not response:
        return

    if "error" in response:
        st.error(f"❌ Error: {response['error']}")
        st.info("💡 Try rephrasing your query or check the logs for more details.")
        return

    # HITL approval UI
    if response.get("awaiting_approval"):
        st.warning("🛑 Research complete. Awaiting your approval to proceed to writing.")
        st.markdown("### 🧾 Research Notes")
        st.text_area("Research Notes", response.get("research_notes", ""), height=300)
        st.markdown("### 📚 Sources")
        st.text_area("Sources", response.get("state", {}).get("sources", ""), height=250)

        col_approve, col_reject = st.columns([1, 1])
        if col_approve.button("✅ Approve and Write Report", type="primary"):
            with st.spinner("✍️ Writing report..."):
                resumed = run_agent(query, graph, config, resume_state=response.get("state", {}), approved=True)
                st.session_state.last_response = resumed
                st.rerun()
        if col_reject.button("🛑 Reject and Stop"):
            st.session_state.last_response = None
            st.info("Approval rejected. You can refine the query and retry.")
        return

    # Display tabs for results
    tabs = st.tabs([
        "🧩 Final Report",
        "🧾 Research Notes",
        "✍️ Draft",
        "✏️ Edited",
        "✅ Fact Check",
        "📚 ArXiv Papers",
        "🌐 Web Results",
        "💬 Message History"
    ])
    with tabs[0]:
        st.markdown("### 🧩 Final Report")
        st.markdown(f'<div class="result-box">{response.get("final_report", "No report generated")}</div>', unsafe_allow_html=True)
        tools_used = response.get("tools_used", [])
        if tools_used:
            st.markdown(f"**Tools Used**: {', '.join(tools_used)}")

    with tabs[1]:
        st.markdown("### 🧾 Research Notes")
        st.text_area("Research Notes", response.get("research_notes", ""), height=400)

    with tabs[2]:
        st.markdown("### ✍️ Draft Report")
        st.text_area("Draft", response.get("draft_report", ""), height=400)

    with tabs[3]:
        st.markdown("### ✏️ Edited Report")
        st.text_area("Edited", response.get("edited_report", ""), height=400)

    with tabs[4]:
        st.markdown("### ✅ Fact-Check Report")
        st.text_area("Fact Check", response.get("fact_check_report", ""), height=400)

    with tabs[5]:
        st.markdown("### 📚 Retrieved ArXiv Documents")
        st.text_area("Documents", response.get("retrieved_docs", "No ArXiv documents retrieved."), height=400)

    with tabs[6]:
        st.markdown("### 🌐 Web Search Results")
        st.text_area("Web Results", response.get("web_results", "No web results retrieved."), height=400)

    with tabs[7]:
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
