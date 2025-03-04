# Q/A AI Assistant

This Streamlit application provides a user-friendly interface for the Agentic RAG Chatbot.

## Features

- Modern chat interface similar to ChatGPT
- Document upload and indexing functionality
- Performance metrics tracking
- Responsive design

## Setup Instructions

1. ensure you have the required dependencies:

```bash
 pip install -r requirements.txt
 ```


1. Ensure your OpenAI API key is set in the .env file:

```bash
OPENAI_API_KEY=your_api_key_here
```

2. Make sure Qdrant is running (if using Docker):

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Access the application in your browser at http://localhost:8501

## Using the Interface

- **Chat**: Type your questions in the input box and click Send
- **Document Indexing**: Upload PDF, TXT, or other supported files using the sidebar uploader
- **Knowledge Base**: View indexed documents and reset the knowledge base when needed
- **Performance Metrics**: Monitor documents indexed, queries processed, and average response time

## User Interface

The interface includes:

- Sidebar for document management
- Main chat panel with message history
- Performance metrics dashboard
- Modern, responsive design elements

## Requirements

- Python 3.9+
- Streamlit
- All dependencies from the main RAG system
- Docker (for running Qdrant)