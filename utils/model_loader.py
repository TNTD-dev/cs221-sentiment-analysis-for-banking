"""
Model Loader Module
Load PhoBERT model and tokenizer with caching for optimal performance
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path


@st.cache_resource
def load_model_and_tokenizer(model_path='models/phobert'):
    """
    Load PhoBERT model and tokenizer with Streamlit caching
    
    Args:
        model_path: Path to saved PhoBERT model
        
    Returns:
        tuple: (model, tokenizer, device)
    """
    try:
        # Check if model path exists
        model_dir = Path(model_path)
        if not model_dir.exists():
            st.error(f"❌ Model not found at {model_path}")
            return None, None, None
        
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        st.info(f"📥 Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        st.info(f"📥 Loading PhoBERT model from {model_path}...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=3
        )
        model.to(device)
        model.eval()
        
        st.success(f"✅ Model loaded successfully on {device}!")
        
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None


def get_label_name(label_id):
    """
    Convert label ID to label name
    
    Args:
        label_id: Integer label (0, 1, 2)
        
    Returns:
        str: Label name
    """
    label_map = {
        0: 'Negative',
        1: 'Neutral',
        2: 'Positive'
    }
    return label_map.get(label_id, 'Unknown')


def get_label_emoji(label_id):
    """
    Get emoji for label
    
    Args:
        label_id: Integer label (0, 1, 2)
        
    Returns:
        str: Emoji
    """
    emoji_map = {
        0: '😞',
        1: '😐',
        2: '😊'
    }
    return emoji_map.get(label_id, '❓')


def get_label_color(label_id):
    """
    Get color for label
    
    Args:
        label_id: Integer label (0, 1, 2)
        
    Returns:
        str: Color name
    """
    color_map = {
        0: 'red',
        1: 'gray',
        2: 'green'
    }
    return color_map.get(label_id, 'black')

