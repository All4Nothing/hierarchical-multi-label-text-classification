# TaxoClass: Hierarchical Multi-Label Text Classification

PyTorch implementation of **TaxoClass** framework from the paper:  
*"TaxoClass: Hierarchical Multi-Label Text Classification Using Only Class Names"*

## 📋 Overview

TaxoClass is a weakly-supervised framework for hierarchical multi-label text classification that only requires:
- Document texts
- Class names (no labeled training data)
- Taxonomy hierarchy structure

### Key Features

✅ **Four-Stage Pipeline:**
1. **Document-Class Similarity**: Using textual entailment (RoBERTa-MNLI)
2. **Core Class Mining**: Top-down candidate selection with confidence scoring
3. **Classifier Training**: BERT + GNN architecture
4. **Self-Training**: Multi-label self-training with KL divergence

✅ **Hierarchy-Aware**: Graph Neural Network encodes taxonomy structure  
✅ **No Labeled Data Required**: Weakly-supervised learning from class names  
✅ **Flexible**: Supports any hierarchical taxonomy

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TaxoClass Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│  Stage 1: Document-Class Similarity (RoBERTa-MNLI)        │
│  Stage 2: Core Class Mining (Top-down + Confidence)       │
│  Stage 3: Classifier Training (BERT + GNN)                │
│  Stage 4: Self-Training (KL Divergence)                   │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
taxoclass/
├── config.py                  # Configuration settings
├── main.py                    # Main pipeline
├── requirements.txt           # Dependencies
├── README.md                  # This file
│
├── data/
│   ├── __init__.py
│   └── loader.py              # Data loading and preprocessing
│
├── models/
│   ├── __init__.py
│   ├── similarity.py          # Stage 1: Similarity calculation
│   ├── core_mining.py         # Stage 2: Core class mining
│   ├── classifier.py          # Stage 3: Classifier (BERT+GNN)
│   └── self_training.py       # Stage 4: Self-training
│
├── utils/
│   ├── __init__.py
│   ├── hierarchy.py           # Taxonomy hierarchy processing
│   └── metrics.py             # Evaluation metrics
│
├── cache/                     # Cached similarity matrices
├── outputs/                   # Output files and metrics
└── saved_models/              # Trained model checkpoints
```

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- torch>=2.0.0
- transformers>=4.30.0
- torch-geometric>=2.3.0
- scikit-learn>=1.3.0
- numpy>=1.24.0
- pandas>=2.0.0
- tqdm>=4.65.0
- networkx>=3.0

## 📊 Data Format

### Input Files

Your data directory should contain:

1. **classes.txt**: Class ID and name mapping
```
0    grocery_gourmet_food
1    meat_poultry
2    jerky
...
```

2. **class_hierarchy.txt**: Parent-child relationships
```
0    1
0    8
1    2
...
```

3. **train/train_corpus.txt**: Training documents
```
0    document text here...
1    another document...
...
```

4. **test/test_corpus.txt**: Test documents (same format)

### Update Configuration

Edit `config.py` to point to your data directory:

```python
DATA_DIR = "../Amazon_products"
```

## 🎯 Usage

### Basic Usage

Run the complete TaxoClass pipeline:

```bash
python main.py
```

This will execute all four stages and evaluate on test data.

### Advanced Usage

#### Using Different Similarity Models

```python
# In main.py, modify Stage 1:

# Option 1: Fast similarity (sentence transformers)
use_fast_similarity = True

# Option 2: Full NLI model (more accurate)
use_fast_similarity = False
```

#### Adjusting Training Parameters

Edit `config.py`:

```python
# Classifier training
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Self-training
SELF_TRAIN_ITERATIONS = 5
SELF_TRAIN_TEMPERATURE = 2.0
SELF_TRAIN_THRESHOLD = 0.5
```

#### Skip Self-Training

```python
# In main.py
run_self_training = False
```

### Stage-by-Stage Execution

You can run individual stages:

```python
from config import Config
from utils.hierarchy import TaxonomyHierarchy
from data.loader import DocumentCorpus
from models.similarity import DocumentClassSimilarity

# Stage 1: Similarity calculation
hierarchy = TaxonomyHierarchy(Config.HIERARCHY_FILE, Config.CLASSES_FILE)
corpus = DocumentCorpus(Config.TRAIN_CORPUS)

similarity_calculator = DocumentClassSimilarity(device="cuda")
similarity_matrix = similarity_calculator.compute_similarity_matrix(
    documents=corpus.get_all_texts(),
    class_names=hierarchy.id_to_name
)
```

## 📈 Evaluation Metrics

TaxoClass reports multiple evaluation metrics:

### Standard Metrics
- **Micro-F1 / Macro-F1**: Standard classification metrics
- **Precision / Recall**: At different thresholds
- **Hamming Loss**: Multi-label classification loss

### Hierarchical Metrics
- **Hierarchical Precision@k**: Considers ancestor classes
- **Hierarchical Recall@k**: Considers ancestor classes
- **Hierarchical F1@k**: Harmonic mean
- **nDCG@k**: Normalized Discounted Cumulative Gain

### Example Output

```
==============================================================
Evaluation Metrics
==============================================================

F1 Scores:
  Micro-F1: 0.6523
  Macro-F1: 0.5847

Precision & Recall:
  Micro-Precision: 0.6891
  Macro-Precision: 0.6234
  Micro-Recall: 0.6189
  Macro-Recall: 0.5512

Hamming Loss: 0.0234

Hierarchical Metrics:

  Top-5:
    H-Precision: 0.7234
    H-Recall: 0.6812
    H-F1: 0.7015
    nDCG: 0.7456
==============================================================
```

## 🔧 Configuration Options

Key configuration parameters in `config.py`:

### Stage 1: Similarity
```python
SIMILARITY_MODEL = "roberta-large-mnli"
SIMILARITY_BATCH_SIZE = 16
HYPOTHESIS_TEMPLATE = "This document is about {class_name}"
```

### Stage 2: Core Mining
```python
CANDIDATE_SELECTION_POWER = 2  # (level+1)^2
CONFIDENCE_THRESHOLD_PERCENTILE = 50  # Median
```

### Stage 3: Classifier
```python
DOC_ENCODER_MODEL = "bert-base-uncased"
EMBEDDING_DIM = 768
GNN_HIDDEN_DIM = 512
GNN_NUM_LAYERS = 3
```

### Stage 4: Self-Training
```python
SELF_TRAIN_ITERATIONS = 5
SELF_TRAIN_TEMPERATURE = 2.0
SELF_TRAIN_THRESHOLD = 0.5
```

## 💡 Tips & Best Practices

### GPU Memory Management

If you encounter OOM errors:

```python
# Reduce batch size
BATCH_SIZE = 16
SIMILARITY_BATCH_SIZE = 8

# Use gradient accumulation
# (implement in trainer if needed)
```

### Improve Performance

1. **Use full NLI model** (Stage 1): More accurate but slower
2. **Increase GNN layers** (Stage 3): Better hierarchy encoding
3. **More self-training iterations** (Stage 4): Better convergence

### Speed Up Training

1. **Use fast similarity** (Stage 1): Sentence transformers
2. **Reduce training epochs** (Stage 3)
3. **Skip self-training** for quick experiments

## 🐛 Troubleshooting

### Common Issues

**Issue**: `CUDA out of memory`  
**Solution**: Reduce batch sizes in `config.py`

**Issue**: Similarity calculation is slow  
**Solution**: Set `use_fast_similarity = True` in `main.py`

**Issue**: Poor performance on small datasets  
**Solution**: Reduce model complexity or use pretrained class embeddings

## 📚 Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{taxoclass,
  title={TaxoClass: Hierarchical Multi-Label Text Classification Using Only Class Names},
  author={...},
  booktitle={...},
  year={2023}
}
```

## 📝 License

This implementation is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Note**: This is an implementation for the TaxoClass framework. Adjust hyperparameters based on your specific dataset and task requirements.

