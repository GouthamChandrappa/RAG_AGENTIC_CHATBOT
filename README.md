# Agentic RAG Chatbot

This repository contains an implementation of an agentic Retrieval-Augmented Generation (RAG) chatbot using CrewAI for agent orchestration, Chonkie for semantic document chunking, and DeepEval for LLM response evaluation.


[qaai.webm](https://github.com/user-attachments/assets/520fe4e7-3a86-44f9-be36-b8361d047a0a)



## Features

- **Agentic RAG Architecture**: Uses CrewAI to create specialized agents for retrieval, generation, and evaluation that collaborate to answer user queries.
- **Semantic Chunking**: Leverages Chonkie for advanced semantic chunking of documents, creating more contextually coherent chunks.
- **Comprehensive Evaluation**: Implements DeepEval for rigorous evaluation of both retrieval and generation quality.
- **Modular Design**: Well-structured codebase with separated concerns for easy extension and maintenance.

## Implementation Details

### Chunking Strategy

The implementation uses Chonkie's semantic chunking to create more coherent and contextually meaningful document chunks:

- **Semantic Chunking**: Instead of fixed-size chunking, the system analyzes document structure and content to create chunks that preserve semantic meaning.
- **Optimal Chunk Size**: Configurable chunk size balances specificity with context preservation (default: 1000 characters).
- **Chunk Overlap**: Configurable overlap between chunks ensures that concepts spanning chunk boundaries are properly captured (default: 200 characters).

### Vector Database Choice

Qdrant was selected as the vector database for these reasons:

- **Performance**: Efficient cosine similarity search for embedding vectors.
- **Filtering Capabilities**: Rich filtering options for metadata-based document retrieval.
- **Easy Deployment**: Simple local deployment for development and testing.
- **Scalability**: Can scale to production workloads when needed.

### Agentic Behavior

The system employs three specialized agents that work collaboratively:

1. **Retrieval Agent**: Analyzes queries, retrieves relevant context, and evaluates the quality of retrieved documents.
2. **Generation Agent**: Synthesizes information from retrieved context to generate accurate and helpful responses.
3. **Evaluation Agent**: Assesses both retrieval and generation quality using DeepEval metrics and provides improvement suggestions.

### Evaluation Metrics

The implementation leverages DeepEval to measure both retrieval and generation quality:

1. **Context Retrieval Metrics**:
   - Contextual Precision: Measures the relevance of retrieved chunks.
   - Contextual Recall: Ensures all necessary information is retrieved.
   - Contextual Relevancy: Evaluates alignment between retrieval results and user query.

2. **Content Generation Metrics**:
   - Answer Relevancy: Assesses if the response addresses the user's query.
   - Faithfulness: Verifies the response is grounded in the retrieved context.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/agentic-rag-chatbot.git
cd agentic-rag-chatbot

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY=your_openai_api_key
# Optionally set QDRANT_URL if not using local instance
# export QDRANT_URL=your_qdrant_url
```

## Usage

### Index Documents

```bash
# Index a single document
python main.py --index data/sample_docs/document.pdf

# Index a directory of documents
python main.py --index data/sample_docs/
```

### Process Queries

```bash
# Process a single query
python main.py --query "What is the main objective of the agentic RAG challenge?"

# Run in interactive mode
python main.py --interactive

# Generate evaluation report
python main.py --eval --output results.json
```

## Project Structure

```
project/
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── data/
│   └── sample_docs/
│       └── place_your_documents_here.txt
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── retrieval_agent.py
│   │   ├── generation_agent.py
│   │   └── evaluation_agent.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── chunker.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── embedder.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── vector_store.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py
│   └── crew/
│       ├── __init__.py
│       └── agentic_rag_crew.py
└── tests/
    ├── __init__.py
    ├── test_chunking.py
    ├── test_retrieval.py
    └── test_evaluation.py
