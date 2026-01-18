import os
import sys
import json
import logging
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try relative import first (when used as module), fall back to absolute
try:
    from .tools import rag_search, web_search
except ImportError:
    from tools import rag_search, web_search
    
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
def load_config(config_path="config.yaml"):
    """Load configuration from a YAML file."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info("Configuration loaded successfully from %s", config_path)
        return config
    except FileNotFoundError:
        logger.error("Configuration file %s not found", config_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Error parsing configuration file %s: %s", config_path, e)
        raise

def load_env():
    """Load environment variables from a .env file."""
    try:
        load_dotenv()
        logger.info("Environment variables loaded successfully")
    except Exception as e:
        logger.error("Error loading .env file: %s", e)
        raise

# ================================
# Agent State
# ================================
class AgentState(TypedDict):
    query: str
    messages: List[dict]
    retrieved_docs: str
    web_results: str
    tools_to_use: List[str]
    iteration: int
    research_notes: str
    research_summary: str
    sources: str
    research_complete: bool
    approval_required: bool
    approved: bool
    awaiting_approval: bool
    draft_report: str
    edited_report: str
    fact_check_report: str
    final_report: str
    final_answer: str
    stage: str

# ================================
# Agent Nodes
# ================================
def classify_tools(query: str) -> List[str]:
    """Classify a query to decide which tools to use."""
    try:
        system_prompt = """You are a research assistant classifier. Given a query, determine which tools to use:
- Use 'rag_search' for queries about specific AI/ML research papers, techniques, or concepts from ArXiv.
- Use 'web_search' for general knowledge, current events, or information not in research papers.
- Use BOTH if the query requires both research papers AND general/current information.

Respond with ONLY a JSON object in this exact format:
{"tools": ["rag_search"]} OR {"tools": ["web_search"]} OR {"tools": ["rag_search", "web_search"]}"""

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ]
        response = llm.invoke(messages)
        try:
            tools_dict = json.loads(response.content.strip())
            tools = tools_dict.get("tools", [])
        except json.JSONDecodeError:
            logger.warning("Failed to parse classifier response, defaulting to both tools")
            tools = ["rag_search", "web_search"]

        logger.info("Query classified: %s -> Tools: %s", query, tools)
        return tools
    except Exception as e:
        logger.error("Error in classify_tools: %s", e)
        return ["web_search"]

def _run_rag_search(query: str, config: dict) -> str:
    rag_conf = config["agent"]["tools"]["rag"]
    return rag_search.invoke({
        "query": query,
        "index_name": config["embeddings"]["pinecone"]["index_name"],
        "namespace": config["embeddings"]["pinecone"]["namespace"],
        "top_k": rag_conf.get("top_k", 5)
    })

def _run_web_search(query: str, config: dict) -> str:
    web_conf = config["agent"]["tools"]["web"]
    provider = web_conf.get("provider", "serpapi").lower()
    max_results = web_conf.get("max_results", 5)
    if provider == "serper":
        return web_search.invoke({
            "query": query,
            "max_results": max_results,
            "provider": "serper"
        })
    return web_search.invoke({
        "query": query,
        "max_results": max_results,
        "provider": "serpapi"
    })

def research_agent(state: AgentState) -> AgentState:
    """Collect sources and synthesize research notes."""
    try:
        query = state["query"]
        config = load_config()

        tools = classify_tools(query)
        state["tools_to_use"] = tools
        state["messages"].append({"role": "system", "content": f"Tools selected: {', '.join(tools)}"})

        retrieved_docs = ""
        web_results = ""
        if "rag_search" in tools:
            retrieved_docs = _run_rag_search(query, config)
        if "web_search" in tools:
            web_results = _run_web_search(query, config)

        state["retrieved_docs"] = retrieved_docs
        state["web_results"] = web_results

        system_prompt = """You are a senior research analyst. Synthesize the provided sources into structured research notes.

Output sections:
1) Research Summary (5-8 bullet points)
2) Key Findings (numbered list)
3) Evidence & Sources (bullets with titles/URLs)
4) Open Questions (bullets)

Be precise and avoid hallucinations. If evidence is insufficient, say so."""

        user_prompt = f"""Topic: {query}

Research Papers (RAG):
{retrieved_docs if retrieved_docs else "No research papers retrieved"}

Web Search Results:
{web_results if web_results else "No web results retrieved"}
"""

        llm = ChatOpenAI(
            model=config["agent"]["llm_model"],
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        state["research_notes"] = response.content
        state["research_summary"] = response.content
        state["sources"] = "\n".join(
            ["From RAG:\n" + (retrieved_docs or "No RAG sources"), "From Web:\n" + (web_results or "No web sources")]
        )
        state["research_complete"] = True
        state["stage"] = "research_complete"
        state["messages"].append({"role": "assistant", "content": state["research_notes"]})
        logger.info("Research agent completed for query: %s", query)
        return state
    except Exception as e:
        logger.error("Error in research_agent: %s", e)
        state["research_notes"] = f"Error during research: {str(e)}"
        state["research_summary"] = state["research_notes"]
        state["stage"] = "research_failed"
        return state

def approval_gate(state: AgentState) -> AgentState:
    """Pause for human approval before writing."""
    if state.get("approval_required") and not state.get("approved"):
        state["awaiting_approval"] = True
        state["stage"] = "awaiting_approval"
        logger.info("Awaiting human approval before writing")
        return state

    state["awaiting_approval"] = False
    state["stage"] = "approved"
    return state

def writer_agent(state: AgentState) -> AgentState:
    """Generate a 2,000-word deep-dive report."""
    try:
        config = load_config()
        report_conf = config["agent"]["report"]
        target_word_count = report_conf.get("target_word_count", 2000)
        sections = report_conf.get("sections", [])

        system_prompt = """You are an expert technical writer. Draft a deep-dive report based on the research notes.
The report must be comprehensive, structured, and approximately the target length.
Include clear headings and cite sources by title or URL in parentheses when relevant."""

        user_prompt = f"""Topic: {state['query']}
Target length: {target_word_count} words
Recommended sections: {sections}

Research Notes:
{state.get('research_notes', '')}
"""

        llm = ChatOpenAI(
            model=config["agent"]["llm_model"],
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        state["draft_report"] = response.content
        state["stage"] = "draft_completed"
        logger.info("Writer agent completed")
        return state
    except Exception as e:
        logger.error("Error in writer_agent: %s", e)
        state["draft_report"] = f"Error during drafting: {str(e)}"
        state["stage"] = "draft_failed"
        return state

def editor_agent(state: AgentState) -> AgentState:
    """Edit and improve clarity and structure."""
    try:
        config = load_config()
        system_prompt = """You are a senior editor. Improve clarity, structure, and flow.
Preserve factual claims and citations. Return the full revised report."""

        user_prompt = f"""Draft Report:
{state.get('draft_report', '')}
"""

        llm = ChatOpenAI(
            model=config["agent"]["llm_model"],
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        state["edited_report"] = response.content
        state["stage"] = "edit_completed"
        logger.info("Editor agent completed")
        return state
    except Exception as e:
        logger.error("Error in editor_agent: %s", e)
        state["edited_report"] = f"Error during editing: {str(e)}"
        state["stage"] = "edit_failed"
        return state

def fact_checker_agent(state: AgentState) -> AgentState:
    """Fact-check the report and flag uncertain claims."""
    try:
        config = load_config()
        system_prompt = """You are a meticulous fact-checker. Identify claims that are unsupported or uncertain.
Return a report with:
1) Verified claims
2) Uncertain or weakly supported claims (with suggested caveats)
3) Missing citations
Do not fabricate sources."""

        user_prompt = f"""Report to fact-check:
{state.get('edited_report', '')}

Sources:
{state.get('sources', '')}
"""

        llm = ChatOpenAI(
            model=config["agent"]["llm_model"],
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        state["fact_check_report"] = response.content
        state["stage"] = "fact_check_completed"
        logger.info("Fact-check agent completed")
        return state
    except Exception as e:
        logger.error("Error in fact_checker_agent: %s", e)
        state["fact_check_report"] = f"Error during fact-checking: {str(e)}"
        state["stage"] = "fact_check_failed"
        return state

def finalize_report(state: AgentState) -> AgentState:
    """Finalize the report with fact-check notes and produce the final output."""
    try:
        config = load_config()
        system_prompt = """You are the lead editor. Produce the final report.
Incorporate fact-check notes and add caveats where needed. Keep citations."""

        user_prompt = f"""Edited Report:
{state.get('edited_report', '')}

Fact-Check Notes:
{state.get('fact_check_report', '')}
"""

        llm = ChatOpenAI(
            model=config["agent"]["llm_model"],
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        state["final_report"] = response.content
        state["final_answer"] = state["final_report"]
        state["stage"] = "finalized"
        logger.info("Final report generated")
        return state
    except Exception as e:
        logger.error("Error in finalize_report: %s", e)
        error_msg = f"Error finalizing report: {str(e)}"
        state["final_report"] = error_msg
        state["final_answer"] = error_msg
        state["stage"] = "finalize_failed"
        return state

# ================================
# Routing Functions
# ================================
def route_from_start(state: AgentState) -> str:
    """Route to research or directly to writing when resuming after approval."""
    if state.get("approved") and state.get("research_complete"):
        return "writer_agent"
    return "research_agent"

def route_after_approval(state: AgentState) -> str:
    if state.get("awaiting_approval"):
        return "awaiting_approval"
    return "writer_agent"

# ================================
# Graph Definition
# ================================
def build_graph() -> StateGraph:
    """Build the multi-agent LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("start_router", lambda s: s)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("approval_gate", approval_gate)
    workflow.add_node("awaiting_approval", lambda s: s)
    workflow.add_node("writer_agent", writer_agent)
    workflow.add_node("editor_agent", editor_agent)
    workflow.add_node("fact_checker_agent", fact_checker_agent)
    workflow.add_node("finalize_report", finalize_report)

    workflow.set_entry_point("start_router")

    workflow.add_conditional_edges(
        "start_router",
        route_from_start,
        {
            "research_agent": "research_agent",
            "writer_agent": "writer_agent"
        }
    )

    workflow.add_edge("research_agent", "approval_gate")
    workflow.add_conditional_edges(
        "approval_gate",
        route_after_approval,
        {
            "awaiting_approval": "awaiting_approval",
            "writer_agent": "writer_agent"
        }
    )

    workflow.add_edge("awaiting_approval", END)
    workflow.add_edge("writer_agent", "editor_agent")
    workflow.add_edge("editor_agent", "fact_checker_agent")
    workflow.add_edge("fact_checker_agent", "finalize_report")
    workflow.add_edge("finalize_report", END)

    return workflow.compile()

# ================================
# Main
# ================================
def main():
    """
    Main function to run the research agent on a sample query.
    
    Returns:
        dict: Agent response with query, retrieved_docs, web_results, and final_answer.
    """
    try:
        load_env()
        config = load_config()

        # Initialize graph
        graph = build_graph()
        logger.info("LangGraph workflow initialized")

        # Sample query
        sample_query = "What are the latest advancements in reinforcement learning?"
        initial_state = {
            "query": sample_query,
            "messages": [{"role": "user", "content": sample_query}],
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

        # Run agent
        logger.info("Starting agent execution for query: %s", sample_query)
        result = graph.invoke(initial_state)
        logger.info("Agent execution completed")

        # Save response
        output_json = config["agent"]["output"]["responses_path"]
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        
        response_data = {
            "query": sample_query,
            "tools_used": result.get("tools_to_use", []),
            "retrieved_docs": result.get("retrieved_docs", ""),
            "web_results": result.get("web_results", ""),
            "research_notes": result.get("research_notes", ""),
            "draft_report": result.get("draft_report", ""),
            "edited_report": result.get("edited_report", ""),
            "fact_check_report": result.get("fact_check_report", ""),
            "final_report": result.get("final_report", ""),
            "final_answer": result.get("final_answer", ""),
            "awaiting_approval": result.get("awaiting_approval", False),
            "stage": result.get("stage", ""),
            "messages": result.get("messages", [])
        }
        
        # Load existing responses if file exists
        existing_responses = []
        if os.path.exists(output_json):
            try:
                with open(output_json, "r", encoding="utf-8") as f:
                    existing_responses = json.load(f)
            except Exception as e:
                logger.warning("Could not load existing responses: %s", e)
        
        # Append new response
        existing_responses.append(response_data)
        
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(existing_responses, f, ensure_ascii=False, indent=4)
        logger.info("Saved agent response to %s", output_json)
        
        # Print results
        print("\n" + "="*80)
        print("MULTI-AGENT RESEARCH TEAM RESPONSE")
        print("="*80)
        print(f"\nQuery: {sample_query}")
        print(f"\nTools Used: {', '.join(result.get('tools_to_use', []))}")
        print(f"\n{'-'*80}")
        print("FINAL REPORT:")
        print(f"{'-'*80}")
        print(result.get("final_answer", "No answer generated"))
        print("="*80 + "\n")

        return response_data

    except Exception as e:
        logger.error("Fatal error in main: %s", e)
        raise

if __name__ == "__main__":
    main()