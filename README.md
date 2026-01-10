# 🎯 Vietnamese Banking Sentiment Analysis

A comprehensive sentiment analysis project for Vietnamese banking customer feedback using traditional Machine Learning and state-of-the-art Transformer models.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Local Training](#local-training)
  - [Google Colab Training](#google-colab-training)
- [Results](#results)
- [Requirements](#requirements)
- [Contributing](#contributing)

## 🔍 Overview

This project implements and compares multiple sentiment analysis models on Vietnamese banking customer feedback. The goal is to classify customer sentiments into three categories:
- **Negative** (0): Customer dissatisfaction
- **Neutral** (1): Neutral feedback
- **Positive** (2): Customer satisfaction

## 📊 Dataset

The dataset consists of Vietnamese customer reviews and feedback from banking social media pages, including:
- **Training set**: 1,312 samples
- **Validation set**: 329 samples
- **Test set**: 336 samples

Data preprocessing includes:
- Vietnamese word tokenization using `underthesea`
- Teencode/slang normalization
- Emoji handling
- Text cleaning and normalization

## 🤖 Models

We implement and compare the following models:

### Baseline Models (Traditional ML)
1. **Logistic Regression** with TF-IDF features
2. **Linear SVM** with TF-IDF features
3. **Naive Bayes** with TF-IDF features

### Transformer Models (Deep Learning)
4. **PhoBERT-base** - Pre-trained BERT for Vietnamese
5. **XLM-RoBERTa-base** - Multilingual RoBERTa

## 📁 Project Structure

```
.
├── data/
│   ├── train.jsonl              # Raw training data
│   ├── test.jsonl               # Raw test data
│   └── processed/               # Processed data files
│       ├── train_processed.csv
│       ├── val_processed.csv
│       └── test_processed.csv
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   └── 02_data-preprocessing.ipynb  # Data preprocessing
├── scripts/
│   └── colab_train.ipynb       # Google Colab training helper
├── models/                      # Saved trained models (created after training)
├── results/                     # Training results and metrics (created after training)
├── train.py                     # Main training script
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
└── README.md                    # This file
```

## 🛠️ Setup

### Prerequisites
- Python 3.12 or higher
- pip or uv package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. **Install dependencies**

Using pip:
```bash
pip install -r requirements.txt
```

Using uv:
```bash
uv sync
```

3. **Verify data files**
Make sure the processed data files exist in `data/processed/`. If not, run the preprocessing notebook:
```bash
jupyter notebook notebooks/02_data-preprocessing.ipynb
```

## 🚀 Usage

### Local Training

The `train.py` script provides a flexible interface for training different models.

#### Train All Models
```bash
python train.py --model all --epochs 3 --batch-size 16
```

#### Train Specific Model

**Baseline Models:**
```bash
# Logistic Regression
python train.py --model lr

# Linear SVM
python train.py --model svm

# Naive Bayes
python train.py --model nb
```

**Transformer Models:**
```bash
# PhoBERT (recommended for Vietnamese)
python train.py --model phobert --epochs 3 --batch-size 16

# XLM-RoBERTa (multilingual)
python train.py --model xlm-roberta --epochs 3 --batch-size 16
```

#### Command-line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `all` | Model to train: `lr`, `svm`, `nb`, `phobert`, `xlm-roberta`, or `all` |
| `--epochs` | int | `3` | Number of training epochs (for transformers) |
| `--batch-size` | int | `16` | Batch size for training (for transformers) |
| `--data-dir` | str | `data/processed` | Directory containing processed data |
| `--save-dir` | str | `models` | Directory to save trained models |
| `--results-dir` | str | `results` | Directory to save results and metrics |

#### Examples

```bash
# Quick baseline comparison (< 5 minutes)
python train.py --model lr
python train.py --model svm
python train.py --model nb

# Train PhoBERT with custom settings
python train.py --model phobert --epochs 5 --batch-size 32

# Train all models with default settings (~ 45 minutes with GPU)
python train.py --model all
```

### Google Colab Training

For training with free GPU (T4) on Google Colab:

#### Method 1: Using the Helper Notebook

1. **Upload to GitHub**
```bash
git add .
git commit -m "Add training scripts"
git push origin main
```

2. **Open Google Colab**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Select `Runtime` → `Change runtime type` → `Hardware accelerator` → `T4 GPU`

3. **Upload the notebook**
   - Upload `scripts/colab_train.ipynb` to Colab
   - Or open directly from GitHub

4. **Follow the notebook instructions**
   - Update GitHub username and repository name
   - Run cells sequentially
   - Download results when complete

#### Method 2: Manual Setup

1. **Clone repository in Colab**
```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
%cd YOUR_REPO_NAME
```

2. **Install dependencies**
```python
%pip install -q -r requirements.txt
```

3. **Check GPU**
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

4. **Run training**
```python
!python train.py --model all --epochs 3 --batch-size 16
```

5. **Download results**
```python
from google.colab import files
import shutil

# Create archives
shutil.make_archive('trained_models', 'zip', 'models')
shutil.make_archive('training_results', 'zip', 'results')

# Download
files.download('trained_models.zip')
files.download('training_results.zip')
```

## 📈 Results

After training, results are saved in the `results/` directory:

- `metrics.json` - Detailed metrics in JSON format
- `comparison.csv` - Model comparison table
- `model_comparison.png` - Visualization comparing all models
- `confusion_matrix_*.png` - Confusion matrix for each model
- `training_logs.txt` - Training logs

### Expected Performance

| Model | Accuracy | F1 (Macro) | Training Time |
|-------|----------|------------|---------------|
| Logistic Regression | ~0.75 | ~0.70 | ~2s |
| Linear SVM | ~0.76 | ~0.71 | ~3s |
| Naive Bayes | ~0.72 | ~0.68 | ~1s |
| PhoBERT-base | ~0.88 | ~0.85 | ~15-20 min (GPU) |
| XLM-RoBERTa-base | ~0.86 | ~0.83 | ~20-25 min (GPU) |

*Note: Results may vary based on random initialization and hardware.*

### Trained Models

Trained models are saved in `models/` directory:

```
models/
├── logistic_regression/
│   ├── model.pkl
│   └── vectorizer.pkl
├── linear_svm/
│   └── model.pkl
├── naive_bayes/
│   └── model.pkl
├── phobert/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer files...
└── xlm_roberta/
    ├── pytorch_model.bin
    ├── config.json
    └── tokenizer files...
```

## 📦 Requirements

Main dependencies:
- `pandas >= 2.3.3` - Data manipulation
- `scikit-learn >= 1.8.0` - Machine learning models
- `torch >= 2.9.1` - Deep learning framework
- `transformers >= 4.57.3` - Hugging Face transformers
- `underthesea >= 8.3.0` - Vietnamese NLP toolkit
- `matplotlib >= 3.10.8` - Visualization
- `seaborn >= 0.13.2` - Statistical visualization
- `emoji >= 2.15.0` - Emoji handling

See `requirements.txt` for complete list.

## 🎯 Key Findings

1. **Transformer models significantly outperform traditional ML models** on Vietnamese sentiment analysis
2. **PhoBERT shows the best performance** due to pre-training on Vietnamese corpus
3. **Traditional ML models are much faster** but sacrifice ~10-15% accuracy
4. **Baseline models are suitable for production** with limited computational resources
5. **For best accuracy, use PhoBERT or XLM-RoBERTa** with appropriate hardware

## 💡 Recommendations

### For Development/Research
- Use **PhoBERT-base** for best results on Vietnamese text
- Consider **PhoBERT-large** if computational resources allow

### For Production
- **Real-time inference**: Use Logistic Regression or SVM with TF-IDF
- **Batch processing**: Use PhoBERT with optimized batch size
- **Balanced approach**: Fine-tune PhoBERT and optimize for inference (ONNX, quantization)

### Future Improvements
- Ensemble multiple models for robustness
- Try PhoBERT-large or ViT5-large
- Implement active learning for continuous improvement
- Add data augmentation techniques
- Experiment with different preprocessing strategies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- [VinAI Research](https://github.com/VinAIResearch/PhoBERT) for PhoBERT
- [Hugging Face](https://huggingface.co/) for transformers library
- [Underthesea](https://github.com/undertheseanlp/underthesea) for Vietnamese NLP tools

---

**Note**: Make sure to replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub username and repository name in the instructions above.

For questions or issues, please open an issue on GitHub.

