import os
import sys
import json
import logging
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated

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
    final_answer: str
    tools_to_use: List[str]
    iteration: int

# ================================
# Agent Nodes
# ================================
def query_classifier(state: AgentState) -> AgentState:
    """
    Classify the query to determine which tools to use.
    
    Args:
        state: Current agent state.
    
    Returns:
        AgentState: Updated state with tool decision.
    """
    try:
        query = state["query"]
        
        system_prompt = """You are a research assistant classifier. Given a query, determine which tools to use:
- Use 'rag_search' for queries about specific AI/ML research papers, techniques, or concepts from ArXiv.
- Use 'web_search' for general knowledge, current events, or information not in research papers.
- Use BOTH if the query requires both research papers AND general/current information.

Respond with ONLY a JSON object in this exact format:
{"tools": ["rag_search"]} OR {"tools": ["web_search"]} OR {"tools": ["rag_search", "web_search"]}

Examples:
Query: "What are the latest advancements in reinforcement learning?"
Response: {"tools": ["rag_search", "web_search"]}

Query: "Explain transformer architecture in deep learning"
Response: {"tools": ["rag_search"]}

Query: "What is the weather today?"
Response: {"tools": ["web_search"]}"""

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ]
        
        response = llm.invoke(messages)
        
        # Parse the response
        try:
            tools_dict = json.loads(response.content.strip())
            tools = tools_dict.get("tools", [])
        except json.JSONDecodeError:
            logger.warning("Failed to parse classifier response, defaulting to both tools")
            tools = ["rag_search", "web_search"]
        
        state["tools_to_use"] = tools
        state["messages"].append({"role": "system", "content": f"Tools selected: {', '.join(tools)}"})
        logger.info("Query classified: %s -> Tools: %s", query, tools)
        
        return state

    except Exception as e:
        logger.error("Error in query_classifier: %s", e)
        state["tools_to_use"] = ["web_search"]  # Fallback to web search
        state["messages"].append({"role": "system", "content": f"Error classifying query: {str(e)}"})
        return state

def rag_search_node(state: AgentState) -> AgentState:
    """
    Execute RAG search using the Pinecone tool.
    
    Args:
        state: Current agent state.
    
    Returns:
        AgentState: Updated state with retrieved documents.
    """
    try:
        query = state["query"]
        
        # Load config inside node
        config = load_config()
        rag_conf = config["agent"]["tools"]["rag"]
        
        result = rag_search.invoke({
            "query": query,
            "index_name": config["embeddings"]["pinecone"]["index_name"],
            "namespace": config["embeddings"]["pinecone"]["namespace"],
            "top_k": rag_conf.get("top_k", 5)
        })
        
        state["retrieved_docs"] = result
        state["messages"].append({"role": "system", "content": f"RAG search completed. Retrieved {result.count('Title:') if result else 0} documents."})
        logger.info("RAG search completed for query: %s", query)
        
        return state

    except Exception as e:
        logger.error("Error in rag_search_node: %s", e)
        state["retrieved_docs"] = f"Error in RAG search: {str(e)}"
        state["messages"].append({"role": "system", "content": f"Error in RAG search: {str(e)}"})
        return state

def web_search_node(state: AgentState) -> AgentState:
    """
    Execute web search using the SerpAPI tool.
    
    Args:
        state: Current agent state.
    
    Returns:
        AgentState: Updated state with web search results.
    """
    try:
        query = state["query"]
        
        # Load config inside node
        config = load_config()
        serpapi_conf = config["agent"]["tools"]["serpapi"]
        
        result = web_search.invoke({
            "query": query,
            "max_results": serpapi_conf.get("max_results", 3)
        })
        
        state["web_results"] = result
        state["messages"].append({"role": "system", "content": f"Web search completed. Retrieved {result.count('Title:') if result else 0} results."})
        logger.info("Web search completed for query: %s", query)
        
        return state

    except Exception as e:
        logger.error("Error in web_search_node: %s", e)
        state["web_results"] = f"Error in web search: {str(e)}"
        state["messages"].append({"role": "system", "content": f"Error in web search: {str(e)}"})
        return state

def generate_answer(state: AgentState) -> AgentState:
    """
    Generate the final answer using the LLM and collected information.
    
    Args:
        state: Current agent state.
    
    Returns:
        AgentState: Updated state with final answer.
    """
    try:
        query = state["query"]
        retrieved_docs = state.get("retrieved_docs", "")
        web_results = state.get("web_results", "")
        
        # Load config inside node
        config = load_config()
        
        system_prompt = """You are a knowledgeable research assistant. Your task is to provide accurate, comprehensive answers based on the information provided.

Guidelines:
1. Synthesize information from both research papers (RAG) and web search results
2. Cite sources by mentioning paper titles or URLs when applicable
3. If information is missing or unclear, acknowledge it
4. Provide structured, well-organized responses
5. Focus on accuracy and relevance"""

        user_prompt = f"""Query: {query}

Research Papers (ArXiv):
{retrieved_docs if retrieved_docs else "No research papers retrieved"}

Web Search Results:
{web_results if web_results else "No web results retrieved"}

Please provide a comprehensive answer to the query based on the above information."""

        llm = ChatOpenAI(
            model=config["agent"]["llm_model"], 
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = llm.invoke(messages)
        state["final_answer"] = response.content
        state["messages"].append({"role": "assistant", "content": state["final_answer"]})
        logger.info("Generated final answer for query: %s", query)
        
        return state

    except Exception as e:
        logger.error("Error in generate_answer: %s", e)
        error_msg = f"I apologize, but I encountered an error while generating the answer: {str(e)}"
        state["final_answer"] = error_msg
        state["messages"].append({"role": "assistant", "content": error_msg})
        return state

# ================================
# Routing Functions
# ================================
def route_after_classifier(state: AgentState) -> str:
    """Route to appropriate tool based on classification."""
    tools = state.get("tools_to_use", [])
    
    if "rag_search" in tools:
        return "rag_search"
    elif "web_search" in tools:
        return "web_search"
    else:
        return "generate_answer"

def route_after_rag(state: AgentState) -> str:
    """Route after RAG search - check if web search is also needed."""
    tools = state.get("tools_to_use", [])
    
    if "web_search" in tools:
        return "web_search"
    else:
        return "generate_answer"

# ================================
# Graph Definition
# ================================
def build_graph() -> StateGraph:
    """
    Build the LangGraph workflow for the research agent.
    
    Returns:
        StateGraph: Compiled LangGraph workflow.
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("query_classifier", query_classifier)
    workflow.add_node("rag_search", rag_search_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate_answer", generate_answer)

    # Set entry point
    workflow.set_entry_point("query_classifier")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "query_classifier",
        route_after_classifier,
        {
            "rag_search": "rag_search",
            "web_search": "web_search",
            "generate_answer": "generate_answer"
        }
    )
    
    workflow.add_conditional_edges(
        "rag_search",
        route_after_rag,
        {
            "web_search": "web_search",
            "generate_answer": "generate_answer"
        }
    )
    
    # Direct edge from web_search to generate_answer
    workflow.add_edge("web_search", "generate_answer")
    
    # End after generating answer
    workflow.add_edge("generate_answer", END)

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
        #sample_query = "Explain transformer architecture"  # Only RAG
        sample_query = "What is the weather today?"  # Only Web Search
        #sample_query = "What are the latest advancements in reinforcement learning?"
        initial_state = {
            "query": sample_query,
            "messages": [{"role": "user", "content": sample_query}],
            "retrieved_docs": "",
            "web_results": "",
            "final_answer": "",
            "tools_to_use": [],
            "iteration": 0
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
            "final_answer": result.get("final_answer", ""),
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
        print("RESEARCH AGENT RESPONSE")
        print("="*80)
        print(f"\nQuery: {sample_query}")
        print(f"\nTools Used: {', '.join(result.get('tools_to_use', []))}")
        print(f"\n{'-'*80}")
        print("FINAL ANSWER:")
        print(f"{'-'*80}")
        print(result.get("final_answer", "No answer generated"))
        print("="*80 + "\n")

        return response_data

    except Exception as e:
        logger.error("Fatal error in main: %s", e)
        raise

if __name__ == "__main__":
    main()