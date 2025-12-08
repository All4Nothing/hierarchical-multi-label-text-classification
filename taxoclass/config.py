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
    # Paper: RoBERTa-large-MNLI for textual entailment
    SIMILARITY_MODEL = "roberta-large-mnli"
    SIMILARITY_BATCH_SIZE = 64  # A6000 can handle larger batches (16 -> 64)
    SIMILARITY_MAX_LENGTH = 128
    HYPOTHESIS_TEMPLATE = "This document is about {class_name}"
    
    # Stage 2: Core Class Mining
    # Paper: Select (level+1)^2 candidates at each level
    CANDIDATE_SELECTION_POWER = 2  # (level+1)^POWER candidates per level (paper uses 2)
    CONFIDENCE_THRESHOLD_PERCENTILE = 60  # Percentile for confidence threshold (paper uses median=50, but 60 is more conservative)
    
    # Stage 3: Classifier Training
    # Paper: BERT-base-uncased (768 dimensions) + GNN
    DOC_ENCODER_MODEL = "bert-base-uncased"
    DOC_MAX_LENGTH = 512  # Increase context length (256 -> 512)
    EMBEDDING_DIM = 768  # BERT-base embedding dimension (paper default)
    GNN_HIDDEN_DIM = 512  # GNN hidden dimension (paper default)
    GNN_NUM_LAYERS = 3  # Number of GNN layers (paper default)
    GNN_DROPOUT = 0.1
    # Control GNN edge direction: True = bidirectional (parent<->child), False = top-down only (parent->child)
    # Paper doesn't specify, but bidirectional is standard for hierarchy-aware GNN
    GNN_BIDIRECTIONAL_EDGES = True  # Enable bidirectional edges for better information flow
    
    LEARNING_RATE = 1e-5  # Lower LR for larger model stability (2e-5 -> 1e-5)
    BATCH_SIZE = 16  # Reduced for memory efficiency (64 -> 16)
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 16 * 4 = 64
    NUM_EPOCHS = 15  # More epochs with larger model (10 -> 15)
    WARMUP_STEPS = 1000  # More warmup steps (500 -> 1000)
    WEIGHT_DECAY = 0.01
    
    # Stage 4: Self-Training
    SELF_TRAIN_ITERATIONS = 3
    SELF_TRAIN_EPOCHS_PER_ITER = 3
    SELF_TRAIN_TEMPERATURE = 2.0 
    SELF_TRAIN_THRESHOLD = 0.8 # Threshold를 높여서 확실한 것만 1로 만듦 (매우 중요)
    SELF_TRAIN_LR = 1e-6 # LR을 매우 낮게 설정하여 파라미터가 튀지 않게 함
    
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
    
    # Pipeline Control: Start from specific stage (1, 2, 3, or 4)
    # Set to None or 1 to run from the beginning
    # Set to 2 to load Stage 1 results and start from Stage 2
    # Set to 3 to load Stage 1 and 2 results and start from Stage 3
    # Set to 4 to load all previous results and start from Stage 4
    START_FROM_STAGE = 2  # None or 1 = start from beginning, 2/3/4 = resume from that stage
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_SAVE_DIR, exist_ok=True)

