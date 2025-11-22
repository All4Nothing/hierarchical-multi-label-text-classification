"""
TaxoClass Configuration
"""
import os

class Config:
    # Data paths
    DATA_DIR = "../project_release/Amazon_products"
    CLASSES_FILE = os.path.join(DATA_DIR, "classes.txt")
    HIERARCHY_FILE = os.path.join(DATA_DIR, "class_hierarchy.txt")
    KEYWORDS_FILE = os.path.join(DATA_DIR, "class_related_keywords.txt")
    TRAIN_CORPUS = os.path.join(DATA_DIR, "train/train_corpus.txt")
    TEST_CORPUS = os.path.join(DATA_DIR, "test/test_corpus.txt")
    
    # Output directories
    OUTPUT_DIR = "./outputs"
    CACHE_DIR = "./cache"
    MODEL_SAVE_DIR = "./saved_models"
    
    # Stage 1: Similarity Calculation
    SIMILARITY_MODEL = "roberta-large-mnli"  # or "microsoft/deberta-large-mnli"
    SIMILARITY_BATCH_SIZE = 16
    SIMILARITY_MAX_LENGTH = 256
    HYPOTHESIS_TEMPLATE = "This document is about {class_name}"
    
    # Stage 2: Core Class Mining
    CANDIDATE_SELECTION_POWER = 2  # (level+1)^2
    CONFIDENCE_THRESHOLD_PERCENTILE = 50  # Median
    
    # Stage 3: Classifier Training
    DOC_ENCODER_MODEL = "bert-base-uncased"
    DOC_MAX_LENGTH = 256
    EMBEDDING_DIM = 768
    GNN_HIDDEN_DIM = 512
    GNN_NUM_LAYERS = 3
    GNN_DROPOUT = 0.1
    
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Stage 4: Self-Training
    SELF_TRAIN_ITERATIONS = 5
    SELF_TRAIN_EPOCHS_PER_ITER = 3
    SELF_TRAIN_TEMPERATURE = 2.0
    SELF_TRAIN_THRESHOLD = 0.5
    SELF_TRAIN_LR = 1e-5
    
    # Device
    DEVICE = "cuda"  # or "cpu", "mps"
    
    # Random seed
    SEED = 42
    
    # Evaluation
    EVAL_BATCH_SIZE = 64
    TOP_K = [1, 3, 5, 10]
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_SAVE_DIR, exist_ok=True)

