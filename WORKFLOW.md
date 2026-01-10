# 🔄 Training Workflow Guide

This document provides a step-by-step guide for the complete workflow from local development to Google Colab training.

## 📋 Quick Start Workflow

### 1️⃣ Local Development & Preparation

```bash
# 1. Make sure your data is processed
# Check that these files exist:
ls data/processed/train_processed.csv
ls data/processed/val_processed.csv
ls data/processed/test_processed.csv

# 2. (Optional) Test training locally with baseline models
python train.py --model lr  # Quick test (< 1 minute)

# 3. Add all files to git
git add .
git commit -m "Add training scripts and processed data"
git push origin main
```

### 2️⃣ Google Colab Setup

1. **Open Google Colab**: https://colab.research.google.com/

2. **Change Runtime to GPU**:
   - Click `Runtime` → `Change runtime type`
   - Hardware accelerator: `T4 GPU`
   - Click `Save`

3. **Upload or Open Notebook**:
   - Option A: Upload `scripts/colab_train.ipynb`
   - Option B: Open from GitHub (File → Open Notebook → GitHub)

### 3️⃣ Training on Colab

#### Update Configuration

In the notebook, find this cell and update:

```python
# 📝 REPLACE THESE WITH YOUR ACTUAL VALUES
GITHUB_USERNAME = "YOUR_USERNAME"  # Your GitHub username
REPO_NAME = "YOUR_REPO_NAME"       # Your repository name
```

#### Run Training

Execute all cells in order:
1. ✅ Check GPU
2. ✅ (Optional) Mount Google Drive
3. ✅ Clone repository
4. ✅ Install dependencies
5. ✅ Verify data files
6. ✅ Train models (choose option A, B, C, or D)
7. ✅ View results
8. ✅ Download results
9. ✅ (Optional) Save to Google Drive

### 4️⃣ Download & Analyze Results

After training completes:

1. **Download Results**: Two zip files will be downloaded
   - `trained_models.zip` - Contains all trained model weights
   - `training_results.zip` - Contains metrics, visualizations, and logs

2. **Extract Files**:
```bash
# On your local machine
unzip trained_models.zip -d models/
unzip training_results.zip -d results/
```

3. **View Results**:
```bash
# View comparison table
cat results/comparison.csv

# Open visualizations
open results/model_comparison.png
open results/confusion_matrix_phobert.png
```

## 🎯 Training Options

### Option A: Train All Models (~45 minutes)
**Best for**: Complete comparison and final results

```python
!python train.py --model all --epochs 3 --batch-size 16
```

**You get**: All 5 models trained (LR, SVM, NB, PhoBERT, XLM-RoBERTa)

### Option B: Train Only PhoBERT (~20 minutes)
**Best for**: Quick high-performance model for Vietnamese

```python
!python train.py --model phobert --epochs 3 --batch-size 16
```

**You get**: Best performing model for Vietnamese sentiment analysis

### Option C: Train Only XLM-RoBERTa (~25 minutes)
**Best for**: Multilingual comparison

```python
!python train.py --model xlm-roberta --epochs 3 --batch-size 16
```

**You get**: Strong multilingual baseline

### Option D: Train Only Baseline Models (~2 minutes)
**Best for**: Quick testing and baseline comparison

```python
!python train.py --model lr
!python train.py --model svm
!python train.py --model nb
```

**You get**: Fast traditional ML baselines

## 🔧 Customization

### Adjust Training Parameters

```python
# More epochs (better performance, longer training)
!python train.py --model phobert --epochs 5 --batch-size 16

# Larger batch size (faster training, more memory)
!python train.py --model phobert --epochs 3 --batch-size 32

# Smaller batch size (less memory, slower training)
!python train.py --model phobert --epochs 3 --batch-size 8
```

### Save to Google Drive

If you want results to persist:

```python
# Mount Drive first
from google.colab import drive
drive.mount('/content/drive')

# After training, copy results
import shutil
drive_dir = '/content/drive/MyDrive/sentiment_analysis_results'
shutil.copytree('models', f'{drive_dir}/models', dirs_exist_ok=True)
shutil.copytree('results', f'{drive_dir}/results', dirs_exist_ok=True)
```

## 📊 Understanding Results

### Metrics Explained

- **Accuracy**: Overall correct predictions / total predictions
- **Precision (Weighted)**: Accuracy weighted by class frequency
- **Recall (Weighted)**: Coverage weighted by class frequency
- **F1 (Weighted)**: Harmonic mean of precision and recall (weighted)
- **F1 (Macro)**: Average F1 across all classes (unweighted)

### Result Files

```
results/
├── metrics.json              # All metrics in JSON format
├── comparison.csv            # Easy-to-read comparison table
├── model_comparison.png      # Visual comparison chart
├── confusion_matrix_*.png    # Confusion matrix for each model
└── training_logs.txt         # Detailed training logs
```

### Model Files

```
models/
├── logistic_regression/
│   ├── model.pkl             # Trained model
│   └── vectorizer.pkl        # TF-IDF vectorizer
├── phobert/
│   ├── pytorch_model.bin     # Model weights
│   ├── config.json           # Model configuration
│   └── tokenizer_config.json # Tokenizer configuration
└── ...
```

## ⚡ Tips & Best Practices

### For Faster Training
1. ✅ Use baseline models first to verify setup
2. ✅ Start with fewer epochs (--epochs 2)
3. ✅ Use larger batch sizes if GPU memory allows
4. ✅ Train one model at a time initially

### For Better Results
1. ✅ Use --epochs 5 for transformer models
2. ✅ Ensure GPU is enabled (check with first cell)
3. ✅ Monitor training logs for convergence
4. ✅ Compare multiple runs with different seeds

### Memory Management
- **T4 GPU has ~15GB**: Can handle batch_size up to 32
- **If OOM error**: Reduce batch_size to 8 or 16
- **For larger models**: Use gradient accumulation

### Troubleshooting

**Problem**: "No GPU found"
- **Solution**: Change runtime type to T4 GPU

**Problem**: "File not found" errors
- **Solution**: Check GitHub username and repo name are correct

**Problem**: "CUDA out of memory"
- **Solution**: Reduce batch size: `--batch-size 8`

**Problem**: Training is very slow
- **Solution**: Verify GPU is active, check GPU usage in Runtime menu

**Problem**: Repository not cloning
- **Solution**: Make sure repository is public or use authentication

## 🚀 Example Complete Workflow

```bash
# === LOCAL (Your Computer) ===
# 1. Verify data
ls data/processed/*.csv

# 2. Test locally (optional)
python train.py --model lr

# 3. Push to GitHub
git add .
git commit -m "Ready for training"
git push origin main

# === GOOGLE COLAB ===
# 4. Open colab_train.ipynb
# 5. Update GitHub username and repo name
# 6. Runtime → Change runtime → T4 GPU
# 7. Run all cells
# 8. Download results when done

# === LOCAL (After Training) ===
# 9. Extract results
unzip trained_models.zip -d models/
unzip training_results.zip -d results/

# 10. View comparison
cat results/comparison.csv
```

## 📚 Additional Resources

- **Hugging Face PhoBERT**: https://huggingface.co/vinai/phobert-base
- **Google Colab Guide**: https://colab.research.google.com/notebooks/intro.ipynb
- **Transformers Documentation**: https://huggingface.co/docs/transformers/

---

Need help? Open an issue on GitHub or check the main README.md for more details.

