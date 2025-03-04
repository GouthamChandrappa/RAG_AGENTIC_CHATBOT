import os
import argparse
import logging
from pathlib import Path

from src.crew.agentic_rag_crew import AgenticRAGCrew
from src.utils.helpers import save_json, load_json, format_time

import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the agentic RAG system.
    """
    parser = argparse.ArgumentParser(description="Agentic RAG Chatbot with CrewAI and Chonkie")
    
    parser.add_argument(
        "--index", 
        help="Path to document or directory to index",
        type=str
    )
    
    parser.add_argument(
        "--query", 
        help="Query to process",
        type=str
    )
    
    parser.add_argument(
        "--interactive", 
        help="Run in interactive mode",
        action="store_true"
    )
    
    parser.add_argument(
        "--eval", 
        help="Generate evaluation report",
        action="store_true"
    )
    
    parser.add_argument(
        "--output", 
        help="Path to save results",
        type=str,
        default="results.json"
    )
    
    args = parser.parse_args()
    
    # Create the RAG system
    rag_system = AgenticRAGCrew(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        use_semantic_chunking=config.SEMANTIC_CHUNKING
    )
    
    # Index documents if specified
    if args.index:
        path = args.index
        
        if os.path.isdir(path):
            logger.info(f"Indexing directory: {path}")
            num_chunks = rag_system.index_directory(path)
            logger.info(f"Indexed {num_chunks} chunks from directory")
        elif os.path.isfile(path):
            logger.info(f"Indexing file: {path}")
            num_chunks = rag_system.index_document(path)
            logger.info(f"Indexed {num_chunks} chunks from file")
        else:
            logger.error(f"Path not found: {path}")
            return
    
    
    # Process a single query if specified
    if args.query:
        logger.info(f"Processing query: {args.query}")
        result = rag_system.process_query_direct(args.query)
    
        print("\n" + "="*50)
        print(f"Query: {result.get('query', '')}")
        print("-"*50)
    
    # Always ensure we have an answer to display
        answer = result.get('answer', '')
        if not answer or answer.strip() == '':
            answer = "No answer could be generated. Please try a different query or check the document content."
    
        print(f"Answer: {answer}")
        print("-"*50)
        print(f"Processing Time: {format_time(result.get('processing_time', 0))}")
        print("="*50 + "\n")
    
        # Save result
        save_json(result, args.output)
        logger.info(f"Result saved to {args.output}")
    
    # Interactive mode
    if args.interactive:
        logger.info("Starting interactive mode. Type 'exit' to quit.")
        
        while True:
            query = input("\nEnter your query: ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                break
                
            result = rag_system.process_query_direct(query)
            
            print("\n" + "="*50)
            print(f"Answer: {result.get('answer', '')}")
            print("-"*50)
            print(f"Processing Time: {format_time(result.get('processing_time', 0))}")
            print("="*50 + "\n")
    
    # Generate evaluation report
    if args.eval:
        logger.info("Generating evaluation report")
        report = rag_system.get_evaluation_report()
        
        eval_output = Path(args.output).with_suffix('.eval.json')
        save_json(report, str(eval_output))
        
        print("\n" + "="*50)
        print("Evaluation Report:")
        print("-"*50)
        if "report" in report:
            print(report["report"])
        else:
            print(report)
        print("="*50 + "\n")
        
        logger.info(f"Evaluation report saved to {eval_output}")

if __name__ == "__main__":
    main()