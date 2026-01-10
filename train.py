import argparse
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import joblib

warnings.filterwarnings('ignore')

# Global constants
LABEL_NAMES = ['negative', 'neutral', 'positive']
RANDOM_SEED = 42


class SentimentDataset(Dataset):
    """PyTorch Dataset for Sentiment Analysis with Transformers"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx]) if hasattr(self.texts, 'iloc') else str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data(data_dir='data/processed_aug'):
    """Load processed training data"""
    print(f"\n📂 Loading data from {data_dir}...")
    
    train_df = pd.read_csv(f'{data_dir}/train_processed.csv')
    val_df = pd.read_csv(f'{data_dir}/val_processed.csv')
    test_df = pd.read_csv(f'{data_dir}/test_processed.csv')
    
    print(f"✅ Train set: {train_df.shape}")
    print(f"✅ Val set: {val_df.shape}")
    print(f"✅ Test set: {test_df.shape}")
    
    # Extract features and labels
    X_train = train_df['text_clean'].fillna('')
    X_val = val_df['text_clean'].fillna('')
    X_test = test_df['text_clean'].fillna('')
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model and return metrics"""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    results = {
        'Model': model_name,
        'Accuracy': float(acc),
        'Precision (Weighted)': float(precision),
        'Recall (Weighted)': float(recall),
        'F1 (Weighted)': float(f1),
        'F1 (Macro)': float(f1_macro),
        'Precision (Macro)': float(precision_macro),
        'Recall (Macro)': float(recall_macro)
    }
    
    return results


def plot_confusion_matrix(y_true, y_pred, model_name, save_dir='results'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Confusion matrix saved to: {save_path}")


def print_classification_report(y_true, y_pred, model_name):
    """Print detailed classification report"""
    print(f"\n{'='*60}")
    print(f"📊 CLASSIFICATION REPORT - {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0))


def train_baseline(model_name, model, X_train, y_train, X_val, y_val, X_test, y_test, 
                   vectorizer, save_dir='models'):
    """Train baseline ML model with TF-IDF"""
    print(f"\n{'='*60}")
    print(f"🤖 Training {model_name}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Fit vectorizer if not already fitted
    if not hasattr(vectorizer, 'vocabulary_'):
        print("   🔄 Creating TF-IDF features...")
        X_train_vec = vectorizer.fit_transform(X_train)
    else:
        X_train_vec = vectorizer.transform(X_train)
    
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    print(f"   🚀 Training {model_name}...")
    model.fit(X_train_vec, y_train)
    
    train_time = time.time() - start_time
    
    # Predictions
    y_pred_test = model.predict(X_test_vec)
    
    # Evaluate
    results = evaluate_model(y_test, y_pred_test, model_name)
    results['Training Time (s)'] = float(train_time)
    
    print(f"✅ {model_name} trained in {train_time:.2f}s")
    print(f"   Test Accuracy: {results['Accuracy']:.4f}")
    print(f"   F1 (Macro): {results['F1 (Macro)']:.4f}")
    
    # Print detailed report
    print_classification_report(y_test, y_pred_test, model_name)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_test, model_name)
    
    # Save model
    model_dir = f'{save_dir}/{model_name.lower().replace(" ", "_")}'
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, f'{model_dir}/model.pkl')
    if model_name == 'Logistic Regression':  # Save vectorizer only once
        joblib.dump(vectorizer, f'{model_dir}/vectorizer.pkl')
    
    print(f"   💾 Model saved to: {model_dir}")
    
    return results


def train_transformer(model_name, model_id, X_train, y_train, X_val, y_val, X_test, y_test,
                      args, save_dir='models'):
    """Train transformer model (PhoBERT or XLM-RoBERTa)"""
    print(f"\n{'='*60}")
    print(f"🤖 Training {model_name}...")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   🖥️  Using device: {device}")
    
    # Load tokenizer and model
    print(f"   📥 Loading tokenizer and model from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=3,
        problem_type="single_label_classification"
    )
    
    # Create datasets
    print("   🔄 Creating datasets...")
    train_dataset = SentimentDataset(X_train, y_train, tokenizer, max_length=256)
    val_dataset = SentimentDataset(X_val, y_val, tokenizer, max_length=256)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer, max_length=256)
    
    # Training arguments
    output_dir = f'./results/{model_name.lower().replace(" ", "_")}_checkpoints'
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir=f'./logs/{model_name.lower().replace(" ", "_")}',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=RANDOM_SEED
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print(f"   🚀 Starting training...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    
    print(f"✅ {model_name} trained in {train_time:.2f}s ({train_time/60:.2f} minutes)")
    
    # Evaluate on test set
    print("   📊 Evaluating on test set...")
    predictions = trainer.predict(test_dataset)
    y_pred_test = np.argmax(predictions.predictions, axis=1)
    
    # Metrics
    results = evaluate_model(y_test, y_pred_test, model_name)
    results['Training Time (s)'] = float(train_time)
    
    print(f"   Test Accuracy: {results['Accuracy']:.4f}")
    print(f"   F1 (Macro): {results['F1 (Macro)']:.4f}")
    
    # Print detailed report
    print_classification_report(y_test, y_pred_test, model_name)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_test, model_name)
    
    # Save model
    model_dir = f'{save_dir}/{model_name.lower().replace(" ", "_")}'
    os.makedirs(model_dir, exist_ok=True)
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    print(f"   💾 Model saved to: {model_dir}")
    
    return results


def save_results(all_results, save_dir='results'):
    """Save all results to JSON and CSV"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as JSON
    json_path = f'{save_dir}/metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Metrics saved to: {json_path}")
    
    # Save as CSV
    df = pd.DataFrame(all_results)
    csv_path = f'{save_dir}/comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"💾 Comparison table saved to: {csv_path}")
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("📊 FINAL COMPARISON - ALL MODELS")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    
    return df


def visualize_comparison(df, save_dir='results'):
    """Create comparison visualizations"""
    print(f"\n📈 Creating comparison visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy Comparison
    ax1 = axes[0, 0]
    sns.barplot(data=df, x='Model', y='Accuracy', palette='viridis', ax=ax1)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', rotation=45)
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.4f')
    
    # 2. F1 Scores Comparison
    ax2 = axes[0, 1]
    f1_comparison = df[['Model', 'F1 (Weighted)', 'F1 (Macro)']].melt(
        id_vars='Model', var_name='Metric', value_name='Score'
    )
    sns.barplot(data=f1_comparison, x='Model', y='Score', hue='Metric', palette='coolwarm', ax=ax2)
    ax2.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='')
    
    # 3. Precision & Recall
    ax3 = axes[1, 0]
    pr_comparison = df[['Model', 'Precision (Weighted)', 'Recall (Weighted)']].melt(
        id_vars='Model', var_name='Metric', value_name='Score'
    )
    sns.barplot(data=pr_comparison, x='Model', y='Score', hue='Metric', palette='Set2', ax=ax3)
    ax3.set_title('Precision & Recall Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(title='')
    
    # 4. Training Time
    ax4 = axes[1, 1]
    sns.barplot(data=df, x='Model', y='Training Time (s)', palette='rocket', ax=ax4)
    ax4.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Time (seconds)')
    ax4.set_xlabel('')
    ax4.tick_params(axis='x', rotation=45)
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%.1f')
    
    plt.tight_layout()
    save_path = f'{save_dir}/model_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison visualization saved to: {save_path}")


def print_conclusions(df):
    """Print conclusions and recommendations"""
    print(f"\n{'='*80}")
    print("🎯 CONCLUSIONS & RECOMMENDATIONS")
    print(f"{'='*80}")
    
    # Find best model by accuracy
    best_acc_idx = df['Accuracy'].idxmax()
    best_model = df.loc[best_acc_idx, 'Model']
    best_acc = df.loc[best_acc_idx, 'Accuracy']
    
    # Find best model by F1 Macro
    best_f1_idx = df['F1 (Macro)'].idxmax()
    best_f1_model = df.loc[best_f1_idx, 'Model']
    best_f1 = df.loc[best_f1_idx, 'F1 (Macro)']
    
    # Find fastest model
    fastest_idx = df['Training Time (s)'].idxmin()
    fastest_model = df.loc[fastest_idx, 'Model']
    fastest_time = df.loc[fastest_idx, 'Training Time (s)']
    
    print(f"\n🏆 BEST MODEL BY ACCURACY:")
    print(f"   Model: {best_model}")
    print(f"   Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    
    print(f"\n🏆 BEST MODEL BY F1-MACRO:")
    print(f"   Model: {best_f1_model}")
    print(f"   F1 (Macro): {best_f1:.4f}")
    
    print(f"\n⚡ FASTEST MODEL:")
    print(f"   Model: {fastest_model}")
    print(f"   Training Time: {fastest_time:.2f}s")
    
    
    print(f"\n✅ Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Train sentiment analysis models on Vietnamese banking data'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['lr', 'svm', 'nb', 'phobert', 'xlm-roberta', 'all'],
        default='all',
        help='Model to train (default: all)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of epochs for transformer models (default: 3)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for transformer models (default: 16)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory containing processed data (default: data/processed)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='models',
        help='Directory to save trained models (default: models)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory to save results (default: results)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 SENTIMENT ANALYSIS TRAINING")
    print("="*80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Configuration:")
    print(f"   Model(s): {args.model}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Save directory: {args.save_dir}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"   🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"   💻 Running on CPU")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.data_dir)
    
    # Initialize results storage
    all_results = []
    
    # Create TF-IDF vectorizer (shared across baseline models)
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )
    
    # Train models based on selection
    models_to_train = []
    
    if args.model == 'all':
        models_to_train = ['lr', 'svm', 'nb', 'phobert', 'xlm-roberta']
    else:
        models_to_train = [args.model]
    
    # Baseline models
    if 'lr' in models_to_train:
        model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        results = train_baseline(
            'Logistic Regression', model,
            X_train, y_train, X_val, y_val, X_test, y_test,
            tfidf_vectorizer, args.save_dir
        )
        all_results.append(results)
    
    if 'svm' in models_to_train:
        model = LinearSVC(
            C=1.0,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            max_iter=2000
        )
        results = train_baseline(
            'Linear SVM', model,
            X_train, y_train, X_val, y_val, X_test, y_test,
            tfidf_vectorizer, args.save_dir
        )
        all_results.append(results)
    
    if 'nb' in models_to_train:
        model = MultinomialNB(alpha=1.0)
        results = train_baseline(
            'Naive Bayes', model,
            X_train, y_train, X_val, y_val, X_test, y_test,
            tfidf_vectorizer, args.save_dir
        )
        all_results.append(results)
    
    # Transformer models
    if 'phobert' in models_to_train:
        results = train_transformer(
            'PhoBERT', 'vinai/phobert-base',
            X_train, y_train, X_val, y_val, X_test, y_test,
            args, args.save_dir
        )
        all_results.append(results)
    
    if 'xlm-roberta' in models_to_train:
        results = train_transformer(
            'XLM-RoBERTa', 'xlm-roberta-base',
            X_train, y_train, X_val, y_val, X_test, y_test,
            args, args.save_dir
        )
        all_results.append(results)
    
    # Save and visualize results
    if all_results:
        df = save_results(all_results, args.results_dir)
        visualize_comparison(df, args.results_dir)
        print_conclusions(df)
    else:
        print("\n⚠️  No models were trained!")
    
    print(f"\n🎉 All done! Check the '{args.save_dir}' and '{args.results_dir}' directories for outputs.")


if __name__ == '__main__':
    main()

