"""
Research Agent - Main Pipeline Runner
Orchestrates the complete workflow: extraction, processing, embedding, and agent execution
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline modules
try:
    from src.data_extraction import main as run_extraction
    from src.data_processing import main as run_processing
    from src.embeddings import main as run_embedding
    from src.agent_graph import main as run_agent, build_graph, load_config, load_env
except ImportError:
    from data_extraction import main as run_extraction
    from data_processing import main as run_processing
    from embeddings import main as run_embedding
    from agent_graph import main as run_agent, build_graph, load_config, load_env

# ================================
# Logging Configuration
# ================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/main_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================
# Pipeline Stages
# ================================

def run_full_pipeline(skip_extraction=False, skip_processing=False, skip_embedding=False):
    """
    Run the complete research agent pipeline.
    
    Args:
        skip_extraction: Skip data extraction (use existing data)
        skip_processing: Skip data processing (use existing processed data)
        skip_embedding: Skip embedding generation (use existing embeddings)
    
    Returns:
        dict: Pipeline execution results
    """
    pipeline_start = datetime.now()
    results = {
        "status": "started",
        "stages": {},
        "start_time": pipeline_start.isoformat(),
        "end_time": None,
        "duration_seconds": None
    }
    
    try:
        logger.info("="*80)
        logger.info("STARTING RESEARCH AGENT PIPELINE")
        logger.info("="*80)
        
        # Stage 1: Data Extraction
        if not skip_extraction:
            logger.info("\n[STAGE 1/4] Data Extraction from ArXiv")
            logger.info("-"*80)
            stage_start = datetime.now()
            try:
                run_extraction()
                results["stages"]["extraction"] = {
                    "status": "success",
                    "duration_seconds": (datetime.now() - stage_start).total_seconds()
                }
                logger.info("✓ Data extraction completed successfully")
            except Exception as e:
                logger.error("✗ Data extraction failed: %s", e)
                results["stages"]["extraction"] = {
                    "status": "failed",
                    "error": str(e),
                    "duration_seconds": (datetime.now() - stage_start).total_seconds()
                }
                raise
        else:
            logger.info("\n[STAGE 1/4] Data Extraction - SKIPPED")
            results["stages"]["extraction"] = {"status": "skipped"}
        
        # Stage 2: Data Processing
        if not skip_processing:
            logger.info("\n[STAGE 2/4] Data Processing and Chunking")
            logger.info("-"*80)
            stage_start = datetime.now()
            try:
                run_processing()
                results["stages"]["processing"] = {
                    "status": "success",
                    "duration_seconds": (datetime.now() - stage_start).total_seconds()
                }
                logger.info("✓ Data processing completed successfully")
            except Exception as e:
                logger.error("✗ Data processing failed: %s", e)
                results["stages"]["processing"] = {
                    "status": "failed",
                    "error": str(e),
                    "duration_seconds": (datetime.now() - stage_start).total_seconds()
                }
                raise
        else:
            logger.info("\n[STAGE 2/4] Data Processing - SKIPPED")
            results["stages"]["processing"] = {"status": "skipped"}
        
        # Stage 3: Embedding Generation
        if not skip_embedding:
            logger.info("\n[STAGE 3/4] Embedding Generation and Indexing")
            logger.info("-"*80)
            stage_start = datetime.now()
            try:
                run_embedding()
                results["stages"]["embedding"] = {
                    "status": "success",
                    "duration_seconds": (datetime.now() - stage_start).total_seconds()
                }
                logger.info("✓ Embedding generation completed successfully")
            except Exception as e:
                logger.error("✗ Embedding generation failed: %s", e)
                results["stages"]["embedding"] = {
                    "status": "failed",
                    "error": str(e),
                    "duration_seconds": (datetime.now() - stage_start).total_seconds()
                }
                raise
        else:
            logger.info("\n[STAGE 3/4] Embedding Generation - SKIPPED")
            results["stages"]["embedding"] = {"status": "skipped"}
        
        # Stage 4: Agent Execution
        logger.info("\n[STAGE 4/4] Agent Execution (Test Query)")
        logger.info("-"*80)
        stage_start = datetime.now()
        try:
            run_agent()
            results["stages"]["agent"] = {
                "status": "success",
                "duration_seconds": (datetime.now() - stage_start).total_seconds()
            }
            logger.info("✓ Agent execution completed successfully")
        except Exception as e:
            logger.error("✗ Agent execution failed: %s", e)
            results["stages"]["agent"] = {
                "status": "failed",
                "error": str(e),
                "duration_seconds": (datetime.now() - stage_start).total_seconds()
            }
            raise
        
        # Pipeline completion
        pipeline_end = datetime.now()
        duration = (pipeline_end - pipeline_start).total_seconds()
        
        results["status"] = "completed"
        results["end_time"] = pipeline_end.isoformat()
        results["duration_seconds"] = duration
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Total Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info("="*80)
        
        return results
        
    except Exception as e:
        pipeline_end = datetime.now()
        duration = (pipeline_end - pipeline_start).total_seconds()
        
        results["status"] = "failed"
        results["error"] = str(e)
        results["end_time"] = pipeline_end.isoformat()
        results["duration_seconds"] = duration
        
        logger.error("\n" + "="*80)
        logger.error("PIPELINE FAILED")
        logger.error("="*80)
        logger.error(f"Error: {e}")
        logger.error(f"Duration before failure: {duration:.2f} seconds")
        logger.error("="*80)
        
        raise

def run_interactive_query():
    """
    Run interactive query mode - allows user to input queries and get answers.
    """
    try:
        logger.info("="*80)
        logger.info("RESEARCH AGENT - INTERACTIVE QUERY MODE")
        logger.info("="*80)
        logger.info("Type 'quit' or 'exit' to stop\n")
        
        # Load environment and initialize graph
        load_env()
        config = load_config()
        graph = build_graph()
        logger.info("Agent initialized successfully\n")
        
        while True:
            # Get user input
            query = input("\n💬 Enter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                logger.info("Exiting interactive mode...")
                break
            
            if not query:
                print("⚠️  Please enter a valid query")
                continue
            
            # Execute query
            print("\n🤔 Processing your query...\n")
            
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
            
            try:
                result = graph.invoke(initial_state)

                if result.get("awaiting_approval"):
                    approval = input("\n🛑 Research complete. Approve to proceed with writing? (y/n): ").strip().lower()
                    if approval in ["y", "yes"]:
                        result["approved"] = True
                        result["awaiting_approval"] = False
                        result["stage"] = "approved"
                        result = graph.invoke(result)
                    else:
                        print("\nReport generation cancelled by user.")
                        continue
                
                # Display results
                print("\n" + "="*80)
                print("FINAL REPORT")
                print("="*80)
                print(result.get("final_report", result.get("final_answer", "No report generated")))
                print("="*80)
                
                tools_used = result.get("tools_to_use", [])
                if tools_used:
                    print(f"\n🔧 Tools Used: {', '.join(tools_used)}")
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"\n❌ Error: {e}")
        
    except Exception as e:
        logger.error(f"Fatal error in interactive mode: {e}")
        raise

def check_prerequisites():
    """
    Check if all prerequisites are met before running the pipeline.
    
    Returns:
        tuple: (bool, list) - (all_ok, list of missing items)
    """
    missing = []
    
    # Check environment variables
    load_env()
    required_env_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    for var in required_env_vars:
        if not os.getenv(var):
            missing.append(f"Environment variable: {var}")
    
    # Check config file
    if not os.path.exists("config.yaml"):
        missing.append("Configuration file: config.yaml")
    
    # Check directories
    os.makedirs("files", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    return len(missing) == 0, missing

def print_status():
    """Print current pipeline status and existing files."""
    logger.info("="*80)
    logger.info("RESEARCH AGENT - STATUS CHECK")
    logger.info("="*80)
    
    files_to_check = {
        "ArXiv Dataset": "files/arxiv_dataset.json",
        "Expanded Dataset": "files/expanded_dataset.json",
        "Embedding Metadata": "files/embedding_metadata.json",
        "Agent Responses": "files/agent_responses.json"
    }
    
    for name, path in files_to_check.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            size_str = f"{size/1024:.2f} KB" if size < 1024*1024 else f"{size/(1024*1024):.2f} MB"
            logger.info(f"✓ {name}: {path} ({size_str})")
        else:
            logger.info(f"✗ {name}: Not found")
    
    logger.info("="*80)

# ================================
# Main Entry Point
# ================================

def main():
    """Main entry point - runs full pipeline by default."""
    try:
        # Check prerequisites
        all_ok, missing = check_prerequisites()
        if not all_ok:
            logger.error("Prerequisites not met:")
            for item in missing:
                logger.error(f"  - Missing: {item}")
            logger.error("\nPlease configure .env file and config.yaml")
            return
        
        # Run full pipeline automatically
        run_full_pipeline(
            skip_extraction=False,
            skip_processing=False,
            skip_embedding=False
        )
        
    except KeyboardInterrupt:
        logger.info("\n\nPipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()