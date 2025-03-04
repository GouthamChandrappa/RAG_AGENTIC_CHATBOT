import os
import logging
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_file_extension(file_path: str) -> str:
    """
    Get the file extension from a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension including the dot (e.g., '.pdf')
    """
    _, ext = os.path.splitext(file_path)
    return ext.lower()

class CustomEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle non-serializable objects."""
    def default(self, obj):
        # Convert CrewOutput objects to a string representation
        if hasattr(obj, 'raw_output'):
            return obj.raw_output
        # Add other special object handling here if needed
        return str(obj)  # Default to string representation for unhandled types

def save_json(data: Dict[str, Any], file_path: str):
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
    """
    try:
        output_file = Path(file_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, cls=CustomEncoder)
            
        logger.info(f"Data saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {str(e)}")
        raise

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, "r") as f:
            data = json.load(f)
            
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise

def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
        
    return text[:max_length] + "..."

def extract_text_from_retrieved_docs(docs: List[Dict[str, Any]]) -> List[str]:
    """
    Extract text content from retrieved documents.
    
    Args:
        docs: List of retrieved documents
        
    Returns:
        List of text content
    """
    return [doc.get("text", "") for doc in docs]