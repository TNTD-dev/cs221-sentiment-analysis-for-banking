# Vietnamese Banking Sentiment Analysis

A comprehensive sentiment analysis project for Vietnamese banking customer feedback using traditional Machine Learning and state-of-the-art Transformer models.

## Features

- **Multiple Models:** Logistic Regression, SVM, Naive Bayes, PhoBERT, XLM-RoBERTa
- **Best Performance:** PhoBERT with 94.61% accuracy
- **Interactive Demo:** Streamlit web app for model testing and error analysis
- **Comprehensive Analysis:** Error analysis, metrics dashboard, sample exploration

## Quick Start

### 1. Train Models

```bash
# Train all models
python train.py 

# Or train specific model
python train.py --model phobert
```

### 2.Run Streamlit App

```bash
# Install dependencies
pip install -r requirements.txt


# Run app
streamlit run app.py
```

Visit: http://localhost:8501

## Model Performance

| Model | Accuracy | F1 (Macro) | Training Time |
|-------|----------|------------|---------------|
| **PhoBERT** | **94.61%** | **63.40%** | ~5 min |
| XLM-RoBERTa | 94.18% | 63.17% | ~25 min |
| Linear SVM | 92.24% | 61.70% | <1 sec |
| Naive Bayes | 92.24% | 61.61% | <1 sec |
| Logistic Regression | 91.16% | 61.08% | ~2 sec |


## Project Structure

```
Final-Project/
├── app.py                      # Streamlit application
├── train.py                    # Model training script
├── utils/                      # Utility modules
│   ├── model_loader.py         # Model loading with caching
│   ├── predictor.py            # Prediction functions
│   └── analyzer.py             # Error analysis functions
├── data/
│   ├── processed/              # Processed datasets
│   └── processed_aug/          # Augmented datasets
├── models/                     # Trained models
│   ├── phobert/
│   ├── xlm-roberta/
│   └── [ML models]/
├── results/                    # Metrics and visualizations
│   ├── metrics.json
│   ├── confusion_matrix_*.png
│   └── comparison.csv
└── notebooks/                  # Jupyter notebooks
    ├── 01_eda.ipynb
    ├── 02_data-preprocessing.ipynb
    ├── 03_ggcolab_train.ipynb
    └── 04_data_augmentation.ipynb
```



## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python 3.10+
- PyTorch 2.9+
- Transformers 4.57+
- Streamlit 1.29+
- Plotly 5.18+
- scikit-learn 1.8+

## Usage Examples

### Training

```bash
# Train all models
python train.py

# Train specific model
python train.py --model phobert --epochs 3 --batch-size 16

# Use augmented data
python train.py --data-dir data/processed_aug
```

### Streamlit App

```bash
# Run app
streamlit run app.py
```

## Results

**Test Set Performance (PhoBERT):**
- Overall Accuracy: **94.61%**
- Negative F1: ~0.60
- Neutral F1: ~0.50 (most challenging)
- Positive F1: ~0.80

**Key Insights:**
- Transformer models significantly outperform traditional ML
- PhoBERT slightly better than XLM-RoBERTa for Vietnamese
- Neutral class is hardest to classify (ambiguous sentiment)
- Model handles informal language and typos well





