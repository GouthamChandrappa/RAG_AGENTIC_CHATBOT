import streamlit as st
import pandas as pd
import os
import tempfile
import time
from pathlib import Path
import PyPDF2
import matplotlib.pyplot as plt
import numpy as np

# Import the RAG system components
from src.crew.agentic_rag_crew import AgenticRAGCrew
from src.utils.helpers import format_time
import config

# Page setup - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title=" Q/A AI - Agentic Document Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4169E1;
        font-weight: bold;
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 1rem;
        color: #333;
    }
    .info-box {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
    }
    .user-message {
        background-color: #2c3e50;
        margin-left: 2rem;
        margin-right: 0.5rem;
        color: white;
    }
    .bot-message {
        background-color: #34495e;
        margin-right: 2rem;
        margin-left: 0.5rem;
        color: white;
    }
    .message-content {
        margin-left: 1rem;
    }
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .user-avatar {
        background-color: #4169E1;
        color: white;
    }
    .bot-avatar {
        background-color: #FF4B4B;
        color: white;
    }
    .stProgress > div > div {
        background-color: #4169E1;
    }
    .sidebar-section {
        margin-bottom: 2rem;
    }
    /* Custom file uploader styling */
    .css-1cpxqw2 {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 20px;
    }
    .css-1cpxqw2:hover {
        border-color: #FF4B4B;
    }
    /* Make the sidebar darker */
    .css-1d391kg {
        background-color: #1E1E1E;
    }
    /* Process button styling */
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 10px 20px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    /* Chat container background */
    .chat-container {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    /* Metrics styling */
    .metric-container {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 1rem;
        font-weight: bold;
        margin-bottom: 5px;
        color: white;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .metric-bar {
        height: 10px;
        border-radius: 5px;
        margin-top: 5px;
    }
    .main-content {
        background-color: #121212;
        color: white;
        padding: 2rem;
        border-radius: 10px;
    }
    /* Chat input styling */
    .stChatInput input {
        border-radius: 20px !important;
        border: 1px solid #333 !important;
        padding: 10px 15px !important;
        background-color: #2c3e50 !important;
        color: white !important;
    }
    .stChatInput button {
        border-radius: 50% !important;
        background-color: #4169E1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'refresh_button' not in st.session_state:
    st.session_state.refresh_button = False
if 'document_text' not in st.session_state:
    st.session_state.document_text = {}
if 'rag_crew' not in st.session_state:
    # Initialize the RAG system with default configuration
    st.session_state.rag_crew = AgenticRAGCrew(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        use_semantic_chunking=config.SEMANTIC_CHUNKING,
        eval_metrics=config.EVAL_METRICS,
        max_iterations=config.MAX_ITERATIONS
    )
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = {
        'contextual_relevancy': 0.0,
        'contextual_precision': 0.0,
        'contextual_recall': 0.0,
        'answer_relevancy': 0.0,
        'faithfulness': 0.0,
        'overall_score': 0.0
    }

# Sidebar
with st.sidebar:
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("## Document Settings")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    st.markdown("Limit 200MB per file • PDF")
    
    # Add an uploaded file preview if a file is uploaded but not processed
    if uploaded_file is not None and not st.session_state.document_processed:
        file_name = uploaded_file.name
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Convert to MB
        st.markdown(f"""
        <div style="padding: 10px; border-radius: 5px; background-color: #2c3e50; margin-top: 10px;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 24px; margin-right: 10px;">📄</span>
                <div>
                    <div style="font-weight: bold;">{file_name}</div>
                    <div style="font-size: 0.8rem; color: #ccc;">{file_size:.2f} MB</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Process and refresh buttons
    col1, col2 = st.columns(2)
    with col1:
        process_button = st.button("Process Document", type="primary")
    with col2:
        refresh_clicked = st.button("Refresh Knowledge", type="secondary")
        
    # Update session state based on button clicks
    if refresh_clicked:
        st.session_state.refresh_button = True
    else:
        st.session_state.refresh_button = False
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add metrics visualization based on the evaluator.py file
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("## Evaluation Metrics")
    
    if st.session_state.document_processed:
        # Overall score with a gauge chart
        overall_score = st.session_state.evaluation_metrics['overall_score']
        
        # Custom HTML/CSS for a circular progress indicator
        st.markdown(f"""
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
            <div style="position: relative; width: 150px; height: 150px;">
                <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; 
                            border-radius: 50%; background: conic-gradient(#4169E1 {overall_score*360}deg, #2c3e50 0deg);"></div>
                <div style="position: absolute; top: 10px; left: 10px; width: calc(100% - 20px); height: calc(100% - 20px); 
                            border-radius: 50%; background-color: #1E1E1E; display: flex; align-items: center; justify-content: center;">
                    <div style="text-align: center;">
                        <div style="font-size: 0.9rem; color: #ccc;">Overall Score</div>
                        <div style="font-size: 2rem; font-weight: bold; color: white;">{overall_score:.2f}</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Individual metrics
        metrics = {
            'Contextual Relevancy': st.session_state.evaluation_metrics['contextual_relevancy'],
            'Contextual Precision': st.session_state.evaluation_metrics['contextual_precision'],
            'Contextual Recall': st.session_state.evaluation_metrics['contextual_recall'],
            'Answer Relevancy': st.session_state.evaluation_metrics['answer_relevancy'],
            'Faithfulness': st.session_state.evaluation_metrics['faithfulness']
        }
        
        for metric_name, metric_value in metrics.items():
            # Determine color based on value
            if metric_value >= 0.8:
                bar_color = "#4CAF50"  # Green for high scores
            elif metric_value >= 0.6:
                bar_color = "#FFC107"  # Yellow for medium scores
            else:
                bar_color = "#F44336"  # Red for low scores
                
            st.markdown(f"""
            <div class="metric-container">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="metric-title">{metric_name}</div>
                    <div style="font-weight: bold; color: {bar_color};">{metric_value:.2f}</div>
                </div>
                <div class="metric-bar" style="width: 100%; background-color: #2c3e50;">
                    <div style="width: {metric_value*100}%; height: 100%; background-color: {bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding: 20px; text-align: center; color: #ccc; background-color: #2c3e50; border-radius: 10px;">
            <div style="font-size: 24px; margin-bottom: 10px;">📊</div>
            <div>Metrics will appear after processing a document</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # About section
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("## About")
    st.markdown("""
    ** Q/A AI** is an agentic document intelligence system that uses Retrieval-Augmented Generation (RAG) to answer questions based on your documents.
    
    **Version:** 1.0.0
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Main area
st.markdown("<div class='main-content'>", unsafe_allow_html=True)
st.markdown("<h1 class='main-header'>Q/A AI: Agentic Document Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Upload a PDF, ask questions, and get accurate answers based on document content.</p>", unsafe_allow_html=True)

# Knowledge base refresh trigger
if st.session_state.refresh_button and st.session_state.document_processed:
    with st.spinner('Refreshing knowledge base...'):
        # Reset the RAG system
        try:
            st.session_state.rag_crew.reset()
            
            # Re-index the current document if available
            if st.session_state.current_file and uploaded_file is not None:
                # Create a temporary file with the document content
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name
                
                # Index the document
                num_chunks = st.session_state.rag_crew.index_document(temp_path)
                
                # Clean up
                os.unlink(temp_path)
                
                st.success(f"Knowledge base refreshed successfully! Indexed {num_chunks} chunks.")
        except Exception as e:
            st.error(f"Error resetting RAG system: {str(e)}")

# Document processing function
if uploaded_file is not None and process_button:
    with st.spinner('Processing document...'):
        # Save the file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        
        # Extract basic file info
        reader = PyPDF2.PdfReader(temp_path)
        num_pages = len(reader.pages)
        
        # Index the document with the RAG system
        progress_bar = st.progress(0)
        
        try:
            # Initialize embedder and vector store with appropriate dimensions
            st.session_state.rag_crew.embedder = st.session_state.rag_crew.embedder.__class__(model_name="all-MiniLM-L6-v2")
            st.session_state.rag_crew.vector_store = st.session_state.rag_crew.vector_store.__class__(embedding_dim=384)
            
            # Chunk and index the document
            progress_bar.progress(0.3)
            num_chunks = st.session_state.rag_crew.index_document(temp_path)
            progress_bar.progress(0.7)
            
            # Extract text for display purposes
            extracted_text = ""
            for i, page in enumerate(reader.pages):
                extracted_text += page.extract_text() + "\n\n"
            
            if 'document_text' not in st.session_state:
                st.session_state.document_text = {}
            
            st.session_state.document_text[uploaded_file.name] = extracted_text
            
            # Store file info
            file_info = {
                'filename': uploaded_file.name,
                'pages': num_pages,
                'size_mb': round(len(uploaded_file.getvalue()) / (1024 * 1024), 2),
                'chunks': num_chunks,
                'text_length': len(extracted_text)
            }
            
            # Update session state
            st.session_state.document_processed = True
            st.session_state.current_file = file_info
            
            st.session_state.evaluation_metrics = {
                metric: 0.0 for metric in config.EVAL_METRICS
            }
            st.session_state.evaluation_metrics['overall_score'] = 0.0


            progress_bar.progress(1.0)
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            # Clear previous chat history when processing a new document
            st.session_state.messages = []
            
            st.success(f"Document processed: {file_info['filename']} ({file_info['pages']} pages, {num_chunks} chunks)")
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            # Clean up on error
            os.unlink(temp_path)

# Document status
if st.session_state.document_processed and st.session_state.current_file:
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    doc_col1, doc_col2, doc_col3 = st.columns([2, 1, 1])
    doc_col1.markdown(f"**Current Document:** {st.session_state.current_file['filename']}")
    doc_col2.markdown(f"**Pages:** {st.session_state.current_file['pages']}")
    doc_col3.markdown(f"**Chunks:** {st.session_state.current_file.get('chunks', 'N/A')}")
    st.markdown("</div>", unsafe_allow_html=True)

# Chat interface
st.markdown("<h2 class='section-header'>Chat with Your Document</h2>", unsafe_allow_html=True)

# Chat container with dark background
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Check if document is processed
if not st.session_state.document_processed:
    st.info("Please upload and process a document first to start chatting.")
else:
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"""
            <div class='chat-message user-message'>
                <div class='avatar user-avatar'>👤</div>
                <div class='message-content'>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='chat-message bot-message'>
                <div class='avatar bot-avatar'>🤖</div>
                <div class='message-content'>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask something about your document..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display the new user message
        st.markdown(f"""
        <div class='chat-message user-message'>
            <div class='avatar user-avatar'>👤</div>
            <div class='message-content'>{prompt}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show typing indicator
        typing_placeholder = st.empty()
        typing_placeholder.markdown("""
        <div class='chat-message bot-message'>
            <div class='avatar bot-avatar'>🤖</div>
            <div class='message-content'>Thinking...</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Process query with the RAG system
        start_time = time.time()
        try:
            # Use direct processing for faster response
            response_data = st.session_state.rag_crew.process_query_direct(prompt)
            
            # Extract the answer from the response
            answer = response_data.get("answer", "I couldn't find an answer to that question in the document.")
            
            # Check if answer is empty or None and provide a fallback
            if not answer or answer.strip() == "":
                answer = "I processed your question but couldn't generate an appropriate response. This might be due to insufficient context in the document."
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Extract evaluation metrics if available
            if "evaluation" in response_data and response_data["evaluation"]:
                eval_metrics = response_data["evaluation"].get("metrics", {})
                # st.write(f"Debug - Raw metrics: {eval_metrics}")
                # Update session metrics if they exist
                for metric_name, value in eval_metrics.items():
                    metric_key = metric_name.lower()
                    if metric_key in st.session_state.evaluation_metrics:
                        st.session_state.evaluation_metrics[metric_key] = value

                st.empty()
                
                # Calculate overall score
                if eval_metrics:
                    overall_score = sum(eval_metrics.values()) / len(eval_metrics)
                    st.session_state.evaluation_metrics['overall_score'] = overall_score

            else:
                st.error("Failed to get evaluation metrics from RAG system")
            
        except Exception as e:
            answer = f"Sorry, I encountered an error when processing your question: {str(e)}"
            processing_time = time.time() - start_time
        
        # Remove typing indicator
        typing_placeholder.empty()
        
        # Add response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer + f"\n\n*Query processing time: {format_time(processing_time)}*"
        })
        
        # Display the new response
        st.markdown(f"""
        <div class='chat-message bot-message'>
            <div class='avatar bot-avatar'>🤖</div>
            <div class='message-content'>{answer}<br><small><i>Query processing time: {format_time(processing_time)}</i></small></div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.8rem; margin-top: 20px;">
    Q/A - Powered by Retrieval-Augmented Generation © 2024
</div>
""", unsafe_allow_html=True)