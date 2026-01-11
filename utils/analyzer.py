"""
Analyzer Module
Functions for loading data, generating predictions, and analyzing errors
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple
from pathlib import Path
import re


@st.cache_data
def load_test_data(data_path: str = 'data/processed_aug/test_processed.csv') -> pd.DataFrame:
    """
    Load test dataset
    
    Args:
        data_path: Path to test data CSV
        
    Returns:
        DataFrame with test data
    """
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"❌ Error loading test data: {str(e)}")
        return pd.DataFrame()


def generate_predictions(df: pd.DataFrame, model, tokenizer, device) -> pd.DataFrame:
    """
    Generate predictions for all samples in the dataframe
    
    Args:
        df: DataFrame with 'text_clean' and 'label' columns
        model: PhoBERT model
        tokenizer: PhoBERT tokenizer
        device: torch device
        
    Returns:
        DataFrame with added prediction columns
    """
    from .predictor import predict_batch
    
    # Get texts
    texts = df['text_clean'].fillna('').tolist()
    
    # Generate predictions
    with st.spinner('🔄 Generating predictions on test set...'):
        predicted_labels, confidence_scores = predict_batch(
            texts, model, tokenizer, device, batch_size=32
        )
    
    # Add to dataframe
    df = df.copy()
    df['predicted_label'] = predicted_labels
    df['confidence_negative'] = [scores[0] for scores in confidence_scores]
    df['confidence_neutral'] = [scores[1] for scores in confidence_scores]
    df['confidence_positive'] = [scores[2] for scores in confidence_scores]
    df['max_confidence'] = [max(scores) for scores in confidence_scores]
    df['is_correct'] = df['label'] == df['predicted_label']
    
    return df


def get_error_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get samples where prediction is incorrect
    
    Args:
        df: DataFrame with predictions
        
    Returns:
        DataFrame with only incorrect predictions
    """
    return df[~df['is_correct']].copy()


def get_correct_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get samples where prediction is correct
    
    Args:
        df: DataFrame with predictions
        
    Returns:
        DataFrame with only correct predictions
    """
    return df[df['is_correct']].copy()


def get_confusion_pairs(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get counts of confusion pairs (true_label -> predicted_label)
    
    Args:
        df: DataFrame with predictions
        
    Returns:
        Dictionary of confusion pairs with counts
    """
    label_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    error_df = get_error_samples(df)
    pairs = {}
    
    for _, row in error_df.iterrows():
        true_label = label_names[row['label']]
        pred_label = label_names[row['predicted_label']]
        pair_key = f"{true_label} → {pred_label}"
        pairs[pair_key] = pairs.get(pair_key, 0) + 1
    
    return pairs


def explain_prediction(text: str, true_label: int, predicted_label: int, 
                       confidence: float, is_correct: bool) -> str:
    """
    Generate a simple explanation for the prediction
    
    Args:
        text: Input text
        true_label: True label
        predicted_label: Predicted label
        confidence: Confidence score (0-1)
        is_correct: Whether prediction is correct
        
    Returns:
        Explanation string
    """
    label_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    # Keywords for sentiment
    negative_keywords = ['không', 'chậm', 'tệ', 'kém', 'lỗi', 'sai', 'tồi', 'khó', 'phí', 'mất']
    positive_keywords = ['tốt', 'nhanh', 'hay', 'ổn', 'ok', 'được', 'tiện', 'dễ', 'thích', 'hài_lòng']
    
    text_lower = text.lower()
    
    if is_correct:
        if confidence >= 0.9:
            return f"✅ Model rất tự tin với **{confidence*100:.1f}%** - Dự đoán chính xác!"
        elif confidence >= 0.7:
            return f"✅ Model tự tin với **{confidence*100:.1f}%** - Dự đoán đúng."
        else:
            return f"✅ Model hơi phân vân (confidence: **{confidence*100:.1f}%**) nhưng vẫn đoán đúng."
    else:
        # Incorrect prediction
        true_name = label_names[true_label]
        pred_name = label_names[predicted_label]
        
        explanation = f"❌ Model dự đoán sai: **{pred_name}** (thực tế: **{true_name}**) với confidence **{confidence*100:.1f}%**."
        
        # Try to provide insight
        neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
        pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
        
        if neg_count > 0 and pos_count > 0:
            explanation += " Text có cả từ tích cực và tiêu cực, gây nhầm lẫn."
        elif neg_count == 0 and pos_count == 0:
            explanation += " Text thiếu từ khóa cảm xúc rõ ràng."
        elif true_label == 1 and predicted_label != 1:
            explanation += " Label Neutral khó phân biệt."
        
        return explanation


def filter_by_label(df: pd.DataFrame, true_label: int = None, 
                    predicted_label: int = None) -> pd.DataFrame:
    """
    Filter dataframe by labels
    
    Args:
        df: DataFrame with predictions
        true_label: Filter by true label (0, 1, 2) or None for all
        predicted_label: Filter by predicted label (0, 1, 2) or None for all
        
    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()
    
    if true_label is not None:
        filtered = filtered[filtered['label'] == true_label]
    
    if predicted_label is not None:
        filtered = filtered[filtered['predicted_label'] == predicted_label]
    
    return filtered


def get_per_class_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate per-class accuracy and metrics
    
    Args:
        df: DataFrame with predictions
        
    Returns:
        DataFrame with per-class metrics
    """
    label_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    metrics = []
    for label_id, label_name in label_names.items():
        class_df = df[df['label'] == label_id]
        total = len(class_df)
        correct = len(class_df[class_df['is_correct']])
        
        if total > 0:
            accuracy = correct / total
            avg_confidence = class_df[class_df['is_correct']]['max_confidence'].mean()
            
            metrics.append({
                'Class': label_name,
                'Total Samples': total,
                'Correct': correct,
                'Incorrect': total - correct,
                'Accuracy': f"{accuracy*100:.2f}%",
                'Avg Confidence (Correct)': f"{avg_confidence*100:.1f}%"
            })
    
    return pd.DataFrame(metrics)


def get_error_statistics(df: pd.DataFrame) -> Dict:
    """
    Get overall error statistics
    
    Args:
        df: DataFrame with predictions
        
    Returns:
        Dictionary with error statistics
    """
    total = len(df)
    correct = len(df[df['is_correct']])
    incorrect = total - correct
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'total_samples': total,
        'correct_predictions': correct,
        'incorrect_predictions': incorrect,
        'accuracy': accuracy,
        'error_rate': 1 - accuracy
    }

