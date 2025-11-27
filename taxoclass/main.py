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

# Wandb import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run 'pip install wandb' to enable logging.")

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
    
    # Initialize wandb
    use_wandb = Config.USE_WANDB and WANDB_AVAILABLE
    if use_wandb:
        # Auto-generate run name if not provided
        run_name = Config.WANDB_RUN_NAME
        if run_name is None:
            run_name = f"taxo_bert-large_gnn{Config.GNN_NUM_LAYERS}_h{Config.GNN_HIDDEN_DIM}"
        
        wandb.init(
            project=Config.WANDB_PROJECT,
            entity=Config.WANDB_ENTITY,
            name=run_name,
            config={
                # Data config
                "seed": Config.SEED,
                "device": device,
                
                # Stage 1 config
                "similarity_model": Config.SIMILARITY_MODEL,
                "similarity_batch_size": Config.SIMILARITY_BATCH_SIZE,
                
                # Stage 2 config
                "candidate_selection_power": Config.CANDIDATE_SELECTION_POWER,
                "confidence_threshold_percentile": Config.CONFIDENCE_THRESHOLD_PERCENTILE,
                
                # Stage 3 config
                "doc_encoder_model": Config.DOC_ENCODER_MODEL,
                "doc_max_length": Config.DOC_MAX_LENGTH,
                "embedding_dim": Config.EMBEDDING_DIM,
                "gnn_hidden_dim": Config.GNN_HIDDEN_DIM,
                "gnn_num_layers": Config.GNN_NUM_LAYERS,
                "gnn_dropout": Config.GNN_DROPOUT,
                "learning_rate": Config.LEARNING_RATE,
                "batch_size": Config.BATCH_SIZE,
                "num_epochs": Config.NUM_EPOCHS,
                "warmup_steps": Config.WARMUP_STEPS,
                "weight_decay": Config.WEIGHT_DECAY,
                
                # Stage 4 config
                "self_train_iterations": Config.SELF_TRAIN_ITERATIONS,
                "self_train_epochs_per_iter": Config.SELF_TRAIN_EPOCHS_PER_ITER,
                "self_train_temperature": Config.SELF_TRAIN_TEMPERATURE,
                "self_train_threshold": Config.SELF_TRAIN_THRESHOLD,
                "self_train_lr": Config.SELF_TRAIN_LR,
                
                # Optimization config
                "use_mixed_precision": Config.USE_MIXED_PRECISION,
                "num_workers": Config.NUM_WORKERS,
                
                # Data usage strategy
                "use_test_in_stage1": Config.USE_TEST_IN_STAGE1,
                "use_test_in_stage2": Config.USE_TEST_IN_STAGE2,
                "use_test_in_stage3": Config.USE_TEST_IN_STAGE3,
                "use_test_in_stage4": Config.USE_TEST_IN_STAGE4,
            },
            tags=Config.WANDB_TAGS
        )
        print(f"\n✅ Wandb initialized: {wandb.run.name}")
        print(f"   URL: {wandb.run.url}")
    elif Config.USE_WANDB and not WANDB_AVAILABLE:
        print("\n⚠️  Wandb requested but not installed. Continuing without logging.")
        print("   Install with: pip install wandb")
    
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
    train_corpus = DocumentCorpus(Config.TRAIN_CORPUS, has_labels=False)
    train_documents = train_corpus.get_all_texts()
    train_labels = train_corpus.get_all_labels()
    
    # Load test corpus
    print("\nLoading test corpus...")
    test_corpus = DocumentCorpus(Config.TEST_CORPUS, has_labels=False)
    test_documents = test_corpus.get_all_texts()
    test_labels = test_corpus.get_all_labels()
    
    # Calculate actual number of classes based on hierarchy
    # Priority: hierarchy.num_classes > max class ID in hierarchy > max label ID
    if hierarchy.id_to_name:
        actual_max_class_id = max(hierarchy.id_to_name.keys())
        # Use hierarchy's num_classes if available, otherwise max_id + 1
        if hasattr(hierarchy, 'num_classes') and hierarchy.num_classes > 0:
            actual_num_classes = max(hierarchy.num_classes, actual_max_class_id + 1)
        else:
            actual_num_classes = actual_max_class_id + 1
    else:
        actual_num_classes = hierarchy.num_classes if hasattr(hierarchy, 'num_classes') else 0
    
    # Validate train and test labels (but don't let them override hierarchy)
    if train_labels:
        train_max_id = max(train_labels)
        if train_max_id >= actual_num_classes:
            # Only warn, don't automatically adjust (hierarchy is source of truth)
            print(f"⚠️  Warning: Train labels contain class ID {train_max_id} >= num_classes ({actual_num_classes})")
            print(f"   This may indicate a data issue. Using hierarchy-based num_classes: {actual_num_classes}")
    
    if test_labels:
        test_max_id = max(test_labels)
        if test_max_id >= actual_num_classes:
            print(f"⚠️  Warning: Test labels contain class ID {test_max_id} >= num_classes ({actual_num_classes})")
            print(f"   This may indicate a data issue. Using hierarchy-based num_classes: {actual_num_classes}")
    
    # Final validation: ensure num_classes is reasonable
    if actual_num_classes > 10000:
        print(f"⚠️  CRITICAL: num_classes ({actual_num_classes}) seems too large!")
        print(f"   This may indicate a calculation error. Using hierarchy.num_classes: {hierarchy.num_classes}")
        actual_num_classes = hierarchy.num_classes
    
    print(f"Using num_classes: {actual_num_classes} (hierarchy.num_classes: {hierarchy.num_classes})")
    
    # Create ground truth matrix for evaluation
    test_ground_truth = create_ground_truth_matrix(
        test_labels,
        hierarchy,
        actual_num_classes
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
            batch_size=Config.SIMILARITY_BATCH_SIZE,
            cache_dir=Config.CACHE_DIR
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
    # Use test data in Stage 1 if configured (transductive learning)
    if Config.USE_TEST_IN_STAGE1:
        print("\n🔄 Stage 1: Using TRAIN + TEST data (transductive learning)")
        stage1_documents = train_documents + test_documents
        print(f"Total documents for similarity: {len(stage1_documents)} (train: {len(train_documents)}, test: {len(test_documents)})")
    else:
        print("\n📊 Stage 1: Using TRAIN data only")
        stage1_documents = train_documents
        print(f"Total documents for similarity: {len(stage1_documents)}")
    
    print("\nComputing similarity matrix...")
    similarity_matrix_all = similarity_calculator.compute_similarity_matrix(
        documents=stage1_documents,
        class_names=hierarchy.id_to_name,
        use_cache=True
    )
    
    # Split similarity matrix back to train/test
    train_similarity_matrix = similarity_matrix_all[:len(train_documents)]
    if Config.USE_TEST_IN_STAGE1:
        test_similarity_matrix = similarity_matrix_all[len(train_documents):]
        print(f"Train similarity matrix shape: {train_similarity_matrix.shape}")
        print(f"Test similarity matrix shape: {test_similarity_matrix.shape}")
    else:
        print(f"Similarity matrix shape: {train_similarity_matrix.shape}")
    
    print(f"Similarity range: [{train_similarity_matrix.min():.4f}, {train_similarity_matrix.max():.4f}]")
    
    # Log to wandb
    if use_wandb:
        wandb.log({
            "stage1/similarity_min": float(train_similarity_matrix.min()),
            "stage1/similarity_max": float(train_similarity_matrix.max()),
            "stage1/similarity_mean": float(train_similarity_matrix.mean()),
            "stage1/similarity_std": float(train_similarity_matrix.std()),
            "stage1/num_documents": len(stage1_documents),
            "stage1/use_test_data": Config.USE_TEST_IN_STAGE1,
        })
    
    # =========================================================================
    # Stage 2: Core Class Mining
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 2: CORE CLASS MINING")
    print("="*80)
    
    # Stage 2: Core Class Mining
    # Use test data if configured
    if Config.USE_TEST_IN_STAGE2 and Config.USE_TEST_IN_STAGE1:
        print("\n🔄 Stage 2: Using TRAIN + TEST data for core class mining")
        stage2_similarity_matrix = similarity_matrix_all
        stage2_documents = stage1_documents
    else:
        print("\n📊 Stage 2: Using TRAIN data only for core class mining")
        stage2_similarity_matrix = train_similarity_matrix
        stage2_documents = train_documents
    
    # Initialize core class miner
    core_miner = CoreClassMiner(
        hierarchy=hierarchy,
        similarity_matrix=stage2_similarity_matrix,
        candidate_power=Config.CANDIDATE_SELECTION_POWER,
        confidence_percentile=Config.CONFIDENCE_THRESHOLD_PERCENTILE
    )
    
    # Identify core classes
    core_classes = core_miner.identify_core_classes()
    
    # Analyze results
    analyzer = CoreClassAnalyzer(core_miner, hierarchy)
    analyzer.print_summary()
    
    # Log to wandb
    if use_wandb:
        # core_classes is {doc_id: core_class_id}
        total_docs_with_core = len(core_classes)
        unique_core_classes = len(set(core_classes.values()))
        avg_docs_per_core_class = total_docs_with_core / unique_core_classes if unique_core_classes > 0 else 0
        
        wandb.log({
            "stage2/num_core_classes": unique_core_classes,
            "stage2/total_docs_with_core": total_docs_with_core,
            "stage2/avg_docs_per_core_class": avg_docs_per_core_class,
            "stage2/num_documents": len(stage2_documents),
            "stage2/use_test_data": Config.USE_TEST_IN_STAGE2 and Config.USE_TEST_IN_STAGE1,
        })
    
    # Filter low confidence documents (optional)
    # core_classes = core_miner.filter_low_confidence_docs(min_confidence=0.0)
    
    # =========================================================================
    # Stage 3: Classifier Training
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 3: CLASSIFIER TRAINING")
    print("="*80)
    
    # Stage 3: Decide which documents to use for training
    if Config.USE_TEST_IN_STAGE3 and Config.USE_TEST_IN_STAGE2 and Config.USE_TEST_IN_STAGE1:
        print("\n🔄 Stage 3: Using TRAIN + TEST data for classifier training")
        stage3_documents = stage2_documents
        stage3_labels = train_labels + test_labels
        
        # Create label matrix for all documents
        # Note: Core classes from Stage 2 already include both train and test
        stage3_label_matrix = create_multi_label_matrix(
            doc_labels=stage3_labels,
            core_class_assignments=core_classes,
            hierarchy=hierarchy,
            num_classes=actual_num_classes
        )
        
        # Split back to train/test for validation split
        train_label_matrix = stage3_label_matrix[:len(train_documents)]
        test_label_matrix_stage3 = stage3_label_matrix[len(train_documents):]
        
        print(f"Total documents for training: {len(stage3_documents)}")
        print(f"  - Train: {len(train_documents)}, Test: {len(test_documents)}")
    else:
        print("\n📊 Stage 3: Using TRAIN data only for classifier training")
        stage3_documents = train_documents
        stage3_labels = train_labels
        
        # Create multi-label training matrix (train only)
        train_label_matrix = create_multi_label_matrix(
            doc_labels=train_labels,
            core_class_assignments=core_classes,
            hierarchy=hierarchy,
            num_classes=actual_num_classes
        )
    
    print(f"\nTraining label matrix shape: {train_label_matrix.shape}")
    print(f"Positive labels: {(train_label_matrix == 1).sum()}")
    print(f"Negative labels: {(train_label_matrix == 0).sum()}")
    print(f"Ignored labels: {(train_label_matrix == -1).sum()}")
    
    # Log to wandb
    if use_wandb:
        wandb.log({
            "stage3/num_positive_labels": int((train_label_matrix == 1).sum()),
            "stage3/num_negative_labels": int((train_label_matrix == 0).sum()),
            "stage3/num_ignored_labels": int((train_label_matrix == -1).sum()),
            "stage3/positive_ratio": float((train_label_matrix == 1).sum() / (train_label_matrix >= 0).sum()),
            "stage3/use_test_data": Config.USE_TEST_IN_STAGE3 and Config.USE_TEST_IN_STAGE2 and Config.USE_TEST_IN_STAGE1,
        })
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(Config.DOC_ENCODER_MODEL)
    
    # Create datasets
    print("\nCreating training dataset...")
    if Config.USE_TEST_IN_STAGE3 and Config.USE_TEST_IN_STAGE2 and Config.USE_TEST_IN_STAGE1:
        # Use all documents (train + test)
        all_label_matrix = stage3_label_matrix
        train_dataset = TaxoDataset(
            documents=stage3_documents,
            labels=all_label_matrix,
            tokenizer=tokenizer,
            max_length=Config.DOC_MAX_LENGTH
        )
    else:
        # Use train only
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
    
    # Log to wandb
    if use_wandb:
        wandb.log({
            "stage3/train_samples": len(train_dataset),
            "stage3/val_samples": len(val_dataset),
        })
    
    # Initialize model
    print("\nInitializing TaxoClassifier...")
    model = TaxoClassifier(
        num_classes=actual_num_classes,  # Use actual_num_classes instead of hierarchy.num_classes
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
    
    # Get edge index for GNN (direction controlled by Config)
    edge_index = torch.LongTensor(
        hierarchy.get_edge_index(bidirectional=Config.GNN_BIDIRECTIONAL_EDGES)
    )
    
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
        save_dir=Config.MODEL_SAVE_DIR,
        use_wandb=use_wandb,
        use_mixed_precision=Config.USE_MIXED_PRECISION,
        gradient_accumulation_steps=getattr(Config, 'GRADIENT_ACCUMULATION_STEPS', 1)
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
        
        # Stage 4: Decide which documents to use for self-training
        if Config.USE_TEST_IN_STAGE4 and Config.USE_TEST_IN_STAGE1:
            print("\n🔄 Stage 4: Using TRAIN + TEST data for self-training")
            stage4_documents = stage1_documents  # train + test
            print(f"Total documents for self-training: {len(stage4_documents)}")
            print(f"  - Train: {len(train_documents)}, Test: {len(test_documents)}")
        else:
            print("\n📊 Stage 4: Using TRAIN data only for self-training")
            stage4_documents = train_documents
            print(f"Total documents for self-training: {len(stage4_documents)}")
        
        # Create unlabeled dataset
        print("\nCreating unlabeled dataset...")
        unlabeled_loader = create_unlabeled_dataset(
            documents=stage4_documents,
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
            save_dir=Config.MODEL_SAVE_DIR,
            use_wandb=use_wandb
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
    
    # Log final metrics to wandb
    if use_wandb:
        wandb_metrics = {f"test/{key}": value for key, value in metrics.items()}
        wandb.log(wandb_metrics)
        
        # Log metrics as summary
        for key, value in metrics.items():
            wandb.run.summary[f"final_{key}"] = value
        
        print("\n✅ Final metrics logged to wandb")
    
    print("\n" + "="*80)
    print("TAXOCLASS PIPELINE COMPLETE!")
    print("="*80)
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()
        print("✅ Wandb run finished")


if __name__ == "__main__":
    main()

