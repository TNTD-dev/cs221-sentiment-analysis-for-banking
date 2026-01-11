"""
Predictor Module
Functions for making predictions with the PhoBERT model
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


def predict_sentiment(text: str, model, tokenizer, device, max_length: int = 256) -> Tuple[int, np.ndarray]:
    """
    Predict sentiment for a single text
    
    Args:
        text: Input text
        model: PhoBERT model
        tokenizer: PhoBERT tokenizer
        device: torch device
        max_length: Maximum sequence length
        
    Returns:
        tuple: (predicted_label, confidence_scores)
    """
    # Tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1)
        confidence_scores = probs.cpu().numpy()[0]
        
        # Get predicted label
        predicted_label = torch.argmax(probs, dim=1).item()
    
    return predicted_label, confidence_scores


def predict_batch(texts: List[str], model, tokenizer, device, max_length: int = 256, batch_size: int = 32) -> Tuple[List[int], List[np.ndarray]]:
    """
    Predict sentiment for multiple texts in batches
    
    Args:
        texts: List of input texts
        model: PhoBERT model
        tokenizer: PhoBERT tokenizer
        device: torch device
        max_length: Maximum sequence length
        batch_size: Batch size for processing
        
    Returns:
        tuple: (predicted_labels, confidence_scores_list)
    """
    model.eval()
    
    all_predictions = []
    all_confidences = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        encodings = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get probabilities
            probs = torch.softmax(logits, dim=1)
            confidence_scores = probs.cpu().numpy()
            
            # Get predicted labels
            predicted_labels = torch.argmax(probs, dim=1).cpu().numpy()
        
        all_predictions.extend(predicted_labels.tolist())
        all_confidences.extend(confidence_scores)
    
    return all_predictions, all_confidences


def format_confidence_dict(confidence_scores: np.ndarray) -> Dict[str, float]:
    """
    Format confidence scores as a dictionary
    
    Args:
        confidence_scores: Numpy array of confidence scores [negative, neutral, positive]
        
    Returns:
        dict: Label names mapped to confidence percentages
    """
    return {
        'Negative': float(confidence_scores[0] * 100),
        'Neutral': float(confidence_scores[1] * 100),
        'Positive': float(confidence_scores[2] * 100)
    }


def get_top_prediction(confidence_scores: np.ndarray) -> Tuple[str, float]:
    """
    Get the top prediction with its confidence
    
    Args:
        confidence_scores: Numpy array of confidence scores
        
    Returns:
        tuple: (label_name, confidence_percentage)
    """
    label_names = ['Negative', 'Neutral', 'Positive']
    top_idx = np.argmax(confidence_scores)
    return label_names[top_idx], float(confidence_scores[top_idx] * 100)

