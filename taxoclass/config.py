"""
TaxoClass Configuration
"""
import os

class Config:
    # Data paths
    DATA_DIR = "../Amazon_products"
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
    SIMILARITY_MODEL = "microsoft/deberta-large-mnli"  # DeBERTa-large for better performance
    SIMILARITY_BATCH_SIZE = 64  # A6000 can handle larger batches (16 -> 64)
    SIMILARITY_MAX_LENGTH = 128
    HYPOTHESIS_TEMPLATE = "This document is about {class_name}"
    
    # Stage 2: Core Class Mining
    CANDIDATE_SELECTION_POWER = 2  # (level+1)^2
    CONFIDENCE_THRESHOLD_PERCENTILE = 50  # Median
    
    # Stage 3: Classifier Training
    DOC_ENCODER_MODEL = "bert-large-uncased"  # Upgrade to bert-large for A6000
    DOC_MAX_LENGTH = 512  # Increase context length (256 -> 512)
    EMBEDDING_DIM = 1024  # bert-large has 1024 dimensions (768 -> 1024)
    GNN_HIDDEN_DIM = 1024  # Scale up GNN hidden dim (512 -> 1024)
    GNN_NUM_LAYERS = 4  # Add one more GNN layer (3 -> 4)
    GNN_DROPOUT = 0.1
    
    LEARNING_RATE = 1e-5  # Lower LR for larger model stability (2e-5 -> 1e-5)
    BATCH_SIZE = 16  # Reduced for memory efficiency (64 -> 16)
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 16 * 4 = 64
    NUM_EPOCHS = 15  # More epochs with larger model (10 -> 15)
    WARMUP_STEPS = 1000  # More warmup steps (500 -> 1000)
    WEIGHT_DECAY = 0.01
    
    # Stage 4: Self-Training
    SELF_TRAIN_ITERATIONS = 5
    SELF_TRAIN_EPOCHS_PER_ITER = 3
    SELF_TRAIN_TEMPERATURE = 2.0
    SELF_TRAIN_THRESHOLD = 0.5
    SELF_TRAIN_LR = 5e-6  # Lower LR for fine-tuning (1e-5 -> 5e-6)
    
    # Data Usage Strategy
    # Transductive learning: Use both train and test data (both are unlabeled)
    USE_TEST_IN_STAGE1 = True   # Zero-shot classification (safe, no label leakage)
    USE_TEST_IN_STAGE2 = True   # Core class mining (safe, confidence-based)
    USE_TEST_IN_STAGE3 = False  # Initial training (conservative, train only)
    USE_TEST_IN_STAGE4 = True   # Self-training (gradual, pseudo-label based)
    
    # Device
    DEVICE = "cuda"  # or "cpu", "mps"
    
    # Random seed
    SEED = 42
    
    # Evaluation
    EVAL_BATCH_SIZE = 128  # Double eval batch size for faster inference (64 -> 128)
    TOP_K = [1, 3, 5, 10]
    
    # A6000 Optimization Settings
    USE_MIXED_PRECISION = True  # Enable FP16/BF16 for faster training
    USE_GRADIENT_CHECKPOINTING = False  # Disable for A6000 (enough memory)
    NUM_WORKERS = 8  # DataLoader workers for faster data loading
    PIN_MEMORY = True  # Pin memory for faster GPU transfer
    
    # Weights & Biases (wandb) Settings
    USE_WANDB = True  # Enable wandb logging
    WANDB_PROJECT = "taxoclass-hierarchical"  # wandb project name
    WANDB_ENTITY = "all4nothing-korea-university"  # wandb entity (team name), None for personal
    WANDB_RUN_NAME = None  # Run name (auto-generated if None)
    WANDB_TAGS = ["hierarchical", "taxonomy", "gnn"]  # Tags for organization
    WANDB_LOG_INTERVAL = 10  # Log every N steps
    WANDB_LOG_GRADIENTS = False  # Log gradient histograms (expensive)
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_SAVE_DIR, exist_ok=True)

