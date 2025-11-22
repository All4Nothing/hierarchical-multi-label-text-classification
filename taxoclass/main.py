"""
TaxoClass Main Pipeline
"""
import os
import sys
import torch
import numpy as np
import random
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from config import Config
from utils.hierarchy import TaxonomyHierarchy
from data.loader import (
    DocumentCorpus,
    TaxoDataset,
    create_multi_label_matrix,
    create_ground_truth_matrix
)
from models.similarity import DocumentClassSimilarity, FastSimilarityCalculator
from models.core_mining import CoreClassMiner, CoreClassAnalyzer
from models.classifier import (
    TaxoClassifier,
    TaxoClassifierTrainer,
    initialize_class_embeddings_with_bert
)
from models.self_training import SelfTrainer, create_unlabeled_dataset
from utils.metrics import evaluate_model


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main TaxoClass pipeline"""
    
    print("="*80)
    print(" "*20 + "TaxoClass Framework")
    print("="*80)
    
    # Set random seed
    set_seed(Config.SEED)
    
    # Create directories
    Config.create_dirs()
    
    # Set device
    device = Config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")
    print(f"Using device: {device}")
    
    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load taxonomy hierarchy
    print("\nLoading taxonomy hierarchy...")
    hierarchy = TaxonomyHierarchy(Config.HIERARCHY_FILE, Config.CLASSES_FILE)
    
    # Load training corpus
    print("\nLoading training corpus...")
    train_corpus = DocumentCorpus(Config.TRAIN_CORPUS)
    train_documents = train_corpus.get_all_texts()
    train_labels = train_corpus.get_all_labels()
    
    # Load test corpus
    print("\nLoading test corpus...")
    test_corpus = DocumentCorpus(Config.TEST_CORPUS)
    test_documents = test_corpus.get_all_texts()
    test_labels = test_corpus.get_all_labels()
    
    # Create ground truth matrix for evaluation
    test_ground_truth = create_ground_truth_matrix(
        test_labels,
        hierarchy,
        hierarchy.num_classes
    )
    
    # =========================================================================
    # Stage 1: Document-Class Similarity Calculation
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 1: DOCUMENT-CLASS SIMILARITY CALCULATION")
    print("="*80)
    
    # Option 1: Use full NLI model (more accurate but slower)
    use_fast_similarity = True  # Set to False for full NLI model
    
    if use_fast_similarity:
        print("\nUsing fast similarity calculator (sentence transformers)...")
        similarity_calculator = FastSimilarityCalculator(
            device=device,
            batch_size=Config.SIMILARITY_BATCH_SIZE
        )
    else:
        print("\nUsing textual entailment model (RoBERTa-MNLI)...")
        similarity_calculator = DocumentClassSimilarity(
            model_name=Config.SIMILARITY_MODEL,
            hypothesis_template=Config.HYPOTHESIS_TEMPLATE,
            device=device,
            batch_size=Config.SIMILARITY_BATCH_SIZE,
            max_length=Config.SIMILARITY_MAX_LENGTH,
            cache_dir=Config.CACHE_DIR
        )
    
    # Compute similarity matrix for training data
    print("\nComputing similarity matrix for training data...")
    train_similarity_matrix = similarity_calculator.compute_similarity_matrix(
        documents=train_documents,
        class_names=hierarchy.id_to_name,
        use_cache=True if hasattr(similarity_calculator, 'cache_dir') else False
    )
    
    print(f"Similarity matrix shape: {train_similarity_matrix.shape}")
    print(f"Similarity range: [{train_similarity_matrix.min():.4f}, {train_similarity_matrix.max():.4f}]")
    
    # =========================================================================
    # Stage 2: Core Class Mining
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 2: CORE CLASS MINING")
    print("="*80)
    
    # Initialize core class miner
    core_miner = CoreClassMiner(
        hierarchy=hierarchy,
        similarity_matrix=train_similarity_matrix,
        candidate_power=Config.CANDIDATE_SELECTION_POWER,
        confidence_percentile=Config.CONFIDENCE_THRESHOLD_PERCENTILE
    )
    
    # Identify core classes
    core_classes = core_miner.identify_core_classes()
    
    # Analyze results
    analyzer = CoreClassAnalyzer(core_miner, hierarchy)
    analyzer.print_summary()
    
    # Filter low confidence documents (optional)
    # core_classes = core_miner.filter_low_confidence_docs(min_confidence=0.0)
    
    # =========================================================================
    # Stage 3: Classifier Training
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 3: CLASSIFIER TRAINING")
    print("="*80)
    
    # Create multi-label training matrix
    print("\nCreating multi-label training matrix...")
    train_label_matrix = create_multi_label_matrix(
        doc_labels=train_labels,
        core_class_assignments=core_classes,
        hierarchy=hierarchy,
        num_classes=hierarchy.num_classes
    )
    
    print(f"Training label matrix shape: {train_label_matrix.shape}")
    print(f"Positive labels: {(train_label_matrix == 1).sum()}")
    print(f"Negative labels: {(train_label_matrix == 0).sum()}")
    print(f"Ignored labels: {(train_label_matrix == -1).sum()}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(Config.DOC_ENCODER_MODEL)
    
    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = TaxoDataset(
        documents=train_documents,
        labels=train_label_matrix,
        tokenizer=tokenizer,
        max_length=Config.DOC_MAX_LENGTH
    )
    
    # Split into train/val (90/10)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    print("\nInitializing TaxoClassifier...")
    model = TaxoClassifier(
        num_classes=hierarchy.num_classes,
        doc_encoder_name=Config.DOC_ENCODER_MODEL,
        embedding_dim=Config.EMBEDDING_DIM,
        gnn_hidden_dim=Config.GNN_HIDDEN_DIM,
        gnn_num_layers=Config.GNN_NUM_LAYERS,
        gnn_dropout=Config.GNN_DROPOUT,
        freeze_bert=False
    )
    
    # Initialize class embeddings with BERT
    print("\nInitializing class embeddings...")
    class_embeddings = initialize_class_embeddings_with_bert(
        class_names=hierarchy.id_to_name,
        bert_model_name=Config.DOC_ENCODER_MODEL,
        device=device
    )
    model.class_embeddings.data = class_embeddings.to(device)
    
    # Get edge index for GNN
    edge_index = torch.LongTensor(hierarchy.get_edge_index())
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = TaxoClassifierTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        edge_index=edge_index,
        device=device,
        learning_rate=Config.LEARNING_RATE,
        num_epochs=Config.NUM_EPOCHS,
        warmup_steps=Config.WARMUP_STEPS,
        weight_decay=Config.WEIGHT_DECAY,
        save_dir=Config.MODEL_SAVE_DIR
    )
    
    # Train model
    print("\nTraining classifier...")
    trainer.train()
    
    # Load best model
    trainer.load_model("best_model.pt")
    
    # =========================================================================
    # Stage 4: Self-Training (Optional)
    # =========================================================================
    run_self_training = True  # Set to False to skip self-training
    
    if run_self_training:
        print("\n" + "="*80)
        print("STAGE 4: SELF-TRAINING")
        print("="*80)
        
        # Create unlabeled dataset (use all training data)
        print("\nCreating unlabeled dataset...")
        unlabeled_loader = create_unlabeled_dataset(
            documents=train_documents,
            tokenizer=tokenizer,
            max_length=Config.DOC_MAX_LENGTH,
            batch_size=Config.BATCH_SIZE
        )
        
        # Initialize self-trainer
        self_trainer = SelfTrainer(
            model=model,
            unlabeled_loader=unlabeled_loader,
            edge_index=edge_index,
            device=device,
            num_iterations=Config.SELF_TRAIN_ITERATIONS,
            num_epochs_per_iter=Config.SELF_TRAIN_EPOCHS_PER_ITER,
            temperature=Config.SELF_TRAIN_TEMPERATURE,
            threshold=Config.SELF_TRAIN_THRESHOLD,
            learning_rate=Config.SELF_TRAIN_LR,
            save_dir=Config.MODEL_SAVE_DIR
        )
        
        # Run self-training
        self_trainer.self_train()
        
        # Load final model
        self_trainer.load_model(f"self_train_iter_{Config.SELF_TRAIN_ITERATIONS}.pt")
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    print("\n" + "="*80)
    print("EVALUATION ON TEST SET")
    print("="*80)
    
    # Create test dataset
    print("\nCreating test dataset...")
    test_dataset = TaxoDataset(
        documents=test_documents,
        labels=test_ground_truth,
        tokenizer=tokenizer,
        max_length=Config.DOC_MAX_LENGTH
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        edge_index=edge_index,
        ground_truth=test_ground_truth,
        hierarchy=hierarchy,
        device=device,
        threshold=0.5,
        top_k_values=Config.TOP_K
    )
    
    # Save metrics
    metrics_file = os.path.join(Config.OUTPUT_DIR, "metrics.txt")
    with open(metrics_file, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"\nMetrics saved to {metrics_file}")
    
    print("\n" + "="*80)
    print("TAXOCLASS PIPELINE COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

