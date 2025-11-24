"""
Generate submission file for Kaggle competition
Loads the best trained model and generates predictions for test set
"""
import os
import csv
from typing import List

import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from utils.hierarchy import TaxonomyHierarchy
from data.loader import DocumentCorpus, TaxoDataset
from models.classifier import (
    TaxoClassifier,
    initialize_class_embeddings_with_bert
)

# Cache for level nodes to avoid recomputation
LEVEL_NODES_CACHE = None


def select_hierarchical_top1(
    probs: np.ndarray,
    level_nodes_cache: dict,
    min_labels: int = 1,
    max_labels: int = None
) -> List[int]:
    """Select one class per level based on highest probability."""
    selected = []
    if not level_nodes_cache:
        return selected
    
    for level in sorted(level_nodes_cache.keys()):
        nodes = level_nodes_cache[level]
        if not nodes:
            continue
        best_node = max(nodes, key=lambda n: probs[n])
        if best_node not in selected:
            selected.append(best_node)
        if max_labels is not None and len(selected) >= max_labels:
            break
    
    if len(selected) < min_labels:
        sorted_indices = np.argsort(probs)[::-1]
        for idx in sorted_indices:
            if idx not in selected:
                selected.append(int(idx))
            if len(selected) >= min_labels:
                break
    return selected


def select_hierarchical_confidence_path(
    probs: np.ndarray,
    level_nodes_cache: dict,
    confidence_threshold: float = 0.5,
    min_labels: int = 1,
    max_labels: int = None
) -> List[int]:
    """
    Select classes along a single hierarchy path, expanding level-by-level
    only if the best class at the next level exceeds confidence_threshold.
    """
    selected = []
    if not level_nodes_cache:
        return selected
    
    for level in sorted(level_nodes_cache.keys()):
        nodes = level_nodes_cache[level]
        if not nodes:
            continue
        best_node = max(nodes, key=lambda n: probs[n])
        best_prob = probs[best_node]
        
        if level == 0 or best_prob >= confidence_threshold:
            if best_node not in selected:
                selected.append(best_node)
        else:
            break
        
        if max_labels is not None and len(selected) >= max_labels:
            break
    
    if len(selected) < min_labels:
        sorted_indices = np.argsort(probs)[::-1]
        for idx in sorted_indices:
            if idx not in selected:
                selected.append(int(idx))
            if len(selected) >= min_labels:
                break
    return selected


def find_best_model(save_dir: str) -> str:
    """
    Find the best model checkpoint to use
    
    Priority:
    1. Self-training final model (self_train_iter_{max_iter}.pt)
    2. Best validation model (best_model.pt)
    3. Latest checkpoint (checkpoint_epoch_{max_epoch}.pt)
    
    Args:
        save_dir: Directory containing saved models
        
    Returns:
        Path to the best model file
    """
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Model directory not found: {save_dir}")
    
    model_files = os.listdir(save_dir)
    
    # Priority 1: Self-training models
    self_train_models = [f for f in model_files if f.startswith("self_train_iter_")]
    if self_train_models:
        # Get the one with highest iteration number
        iterations = [int(f.replace("self_train_iter_", "").replace(".pt", "")) for f in self_train_models]
        max_iter = max(iterations)
        best_model = f"self_train_iter_{max_iter}.pt"
        print(f"✅ Found self-training model: {best_model} (iteration {max_iter})")
        return os.path.join(save_dir, best_model)
    
    # Priority 2: Best validation model
    if "best_model.pt" in model_files:
        print("✅ Found best validation model: best_model.pt")
        return os.path.join(save_dir, "best_model.pt")
    
    # Priority 3: Latest checkpoint
    checkpoint_models = [f for f in model_files if f.startswith("checkpoint_epoch_")]
    if checkpoint_models:
        epochs = [int(f.replace("checkpoint_epoch_", "").replace(".pt", "")) for f in checkpoint_models]
        max_epoch = max(epochs)
        best_model = f"checkpoint_epoch_{max_epoch}.pt"
        print(f"✅ Found latest checkpoint: {best_model} (epoch {max_epoch})")
        return os.path.join(save_dir, best_model)
    
    raise FileNotFoundError(f"No model checkpoint found in {save_dir}")


def load_model(model_path: str, hierarchy: TaxonomyHierarchy, device: str) -> TaxoClassifier:
    """
    Load trained model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        hierarchy: TaxonomyHierarchy object
        device: Device to load model on
        
    Returns:
        Loaded TaxoClassifier model
    """
    print(f"\nLoading model from {model_path}...")
    
    # Calculate num_classes
    if hierarchy.id_to_name:
        actual_max_class_id = max(hierarchy.id_to_name.keys())
        if hasattr(hierarchy, 'num_classes') and hierarchy.num_classes > 0:
            num_classes = max(hierarchy.num_classes, actual_max_class_id + 1)
        else:
            num_classes = actual_max_class_id + 1
    else:
        num_classes = hierarchy.num_classes if hasattr(hierarchy, 'num_classes') else 0
    
    # Initialize model
    model = TaxoClassifier(
        num_classes=num_classes,
        doc_encoder_name=Config.DOC_ENCODER_MODEL,
        embedding_dim=Config.EMBEDDING_DIM,
        gnn_hidden_dim=Config.GNN_HIDDEN_DIM,
        gnn_num_layers=Config.GNN_NUM_LAYERS,
        gnn_dropout=Config.GNN_DROPOUT,
        freeze_bert=False
    )
    
    # Initialize class embeddings with BERT
    print("Initializing class embeddings...")
    class_embeddings = initialize_class_embeddings_with_bert(
        class_names=hierarchy.id_to_name,
        bert_model_name=Config.DOC_ENCODER_MODEL,
        device=device
    )
    model.class_embeddings.data = class_embeddings.to(device)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Debug: print checkpoint keys
    print(f"   Checkpoint keys: {list(checkpoint.keys())}")
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("   Loaded from 'model_state_dict'")
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("   Loaded from 'state_dict'")
    else:
        # Self-training models might only have state_dict at root
        model.load_state_dict(checkpoint)
        print("   Loaded from root level (self-training format)")
    
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    
    return model


def predict_classes(
    model: TaxoClassifier,
    test_loader: DataLoader,
    edge_index: torch.Tensor,
    device: str,
    threshold: float = 0.5,
    top_k: int = None
) -> np.ndarray:
    """
    Generate predictions for test set
    
    Args:
        model: Trained TaxoClassifier model
        test_loader: Test data loader
        edge_index: Hierarchy edge index
        device: Device
        threshold: Probability threshold for binary classification
        top_k: If specified, return top-k classes instead of threshold-based
        
    Returns:
        Probability predictions array (num_samples, num_classes)
    """
    all_predictions = []
    
    print("\nGenerating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get probabilities
            predictions = model(input_ids, attention_mask, edge_index, return_probs=True)
            all_predictions.append(predictions.cpu().numpy())
    
    predictions = np.vstack(all_predictions)
    return predictions


def generate_submission(
    model_path: str = None,
    test_corpus_path: str = None,
    output_path: str = "submission.csv",
    threshold: float = 0.5,
    top_k: int = None,
    min_labels: int = 1,
    max_labels: int = None,  # No limit by default
    use_hierarchical_top1: bool = False,
    use_hierarchical_confidence: bool = False,
    confidence_threshold: float = 0.5
):
    """
    Generate submission file for Kaggle competition
    
    Args:
        model_path: Path to model checkpoint (if None, auto-detect best model)
        test_corpus_path: Path to test corpus file
        output_path: Output submission file path
        threshold: Probability threshold for predictions
        top_k: If specified, use top-k instead of threshold
        min_labels: Minimum number of labels per sample
        max_labels: Maximum number of labels per sample
    """
    print("="*80)
    print(" "*25 + "GENERATE SUBMISSION FILE")
    print("="*80)
    
    # Set device
    device = Config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")
    print(f"Using device: {device}")
    
    # Load hierarchy
    print("\nLoading taxonomy hierarchy...")
    hierarchy = TaxonomyHierarchy(Config.HIERARCHY_FILE, Config.CLASSES_FILE)
    
    # Find best model
    if model_path is None:
        model_path = find_best_model(Config.MODEL_SAVE_DIR)
    else:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model
    model = load_model(model_path, hierarchy, device)
    
    # Get edge index
    edge_index = torch.LongTensor(hierarchy.get_edge_index()).to(device)
    
    # Load test corpus
    if test_corpus_path is None:
        test_corpus_path = Config.TEST_CORPUS
    
    print(f"\nLoading test corpus from {test_corpus_path}...")
    # Test corpus format: pid \t text (not label \t text)
    test_pids = []
    test_documents = []
    with open(test_corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                pid, text = parts
                test_pids.append(pid)
                test_documents.append(text)
    
    print(f"Loaded {len(test_documents)} test documents")
    
    """max_samples = 200
    test_pids = test_pids[:max_samples]
    test_documents = test_documents[:max_samples]"""

    # Create dataset and loader
    print("\nCreating test dataset...")
    tokenizer = BertTokenizer.from_pretrained(Config.DOC_ENCODER_MODEL)
    
    # Create dummy labels (not used for prediction)
    dummy_labels = np.zeros((len(test_documents), hierarchy.num_classes))
    test_dataset = TaxoDataset(
        documents=test_documents,
        labels=dummy_labels,
        tokenizer=tokenizer,
        max_length=Config.DOC_MAX_LENGTH
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Generate predictions
    predictions = predict_classes(
        model=model,
        test_loader=test_loader,
        edge_index=edge_index,
        device=device,
        threshold=threshold,
        top_k=top_k
    )
    
    # Print prediction statistics
    print(f"\n📊 Prediction Statistics:")
    print(f"   Prediction shape: {predictions.shape}")
    print(f"   Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"   Prediction mean: {predictions.mean():.4f}")
    print(f"   Prediction std: {predictions.std():.4f}")
    
    # Check threshold-based predictions
    threshold_predictions = (predictions >= threshold).astype(int)
    num_labels_per_sample = threshold_predictions.sum(axis=1)
    print(f"   Threshold-based: min_labels={num_labels_per_sample.min()}, "
          f"max_labels={num_labels_per_sample.max()}, "
          f"avg_labels={num_labels_per_sample.mean():.2f}")
    
    # Check if predictions are diverse
    sample_predictions = predictions[:5]  # First 5 samples
    for i in range(min(5, len(sample_predictions))):
        top_5 = np.argsort(sample_predictions[i])[-5:][::-1]
        top_5_probs = sample_predictions[i][top_5]
        print(f"   Sample {i} top-5 classes: {top_5.tolist()}, probs: {top_5_probs}")
    
    # Prepare level nodes cache for hierarchical selection
    global LEVEL_NODES_CACHE
    if use_hierarchical_top1 or use_hierarchical_confidence:
        LEVEL_NODES_CACHE = {}
        for node, level in hierarchy.levels.items():
            LEVEL_NODES_CACHE.setdefault(level, []).append(node)
    
    # Convert predictions to label lists
    print("\nConverting predictions to submission format...")
    all_pids = []
    all_labels = []
    
    for i, pid in enumerate(tqdm(test_pids, desc="Formatting predictions")):
        probs = predictions[i]
        
        # Apply selection strategy
        if use_hierarchical_confidence:
            predicted_classes = select_hierarchical_confidence_path(
                probs,
                level_nodes_cache=LEVEL_NODES_CACHE,
                confidence_threshold=confidence_threshold,
                min_labels=min_labels,
                max_labels=max_labels
            )
        elif use_hierarchical_top1:
            predicted_classes = select_hierarchical_top1(
                probs,
                level_nodes_cache=LEVEL_NODES_CACHE,
                min_labels=min_labels,
                max_labels=max_labels
            )
        elif top_k is not None:
            # Get top-k classes
            top_k_indices = np.argsort(probs)[-top_k:][::-1]
            predicted_classes = top_k_indices.tolist()
        else:
            # Threshold-based
            predicted_classes = np.where(probs >= threshold)[0].tolist()
        
        # Ensure min/max labels constraint
        if len(predicted_classes) < min_labels:
            # If too few, add more top classes by probability
            top_indices = np.argsort(probs)[-min_labels:][::-1]
            predicted_classes = [int(idx) for idx in top_indices]
        
        if max_labels is not None and len(predicted_classes) > max_labels:
            # If too many, keep top classes by probability
            # Get probabilities for predicted classes and sort
            class_probs = [(idx, probs[idx]) for idx in predicted_classes]
            class_probs.sort(key=lambda x: x[1], reverse=True)
            predicted_classes = [idx for idx, _ in class_probs[:max_labels]]
        
        # Sort class IDs
        predicted_classes = sorted(predicted_classes)
        
        all_pids.append(pid)
        all_labels.append(predicted_classes)
    
    # Save submission file
    print(f"\nSaving submission file to {output_path}...")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'labels'])
        for pid, labels in zip(all_pids, all_labels):
            labels_str = ','.join(map(str, labels))
            writer.writerow([pid, labels_str])
    
    # Print statistics
    num_labels_per_sample = [len(labels) for labels in all_labels]
    print(f"\n✅ Submission file saved: {output_path}")
    print(f"   Total samples: {len(all_pids)}")
    print(f"   Labels per sample: min={min(num_labels_per_sample)}, "
          f"max={max(num_labels_per_sample)}, "
          f"avg={np.mean(num_labels_per_sample):.2f}")
    print(f"   Total unique classes predicted: {len(set([c for labels in all_labels for c in labels]))}")
    
    print("\n" + "="*80)
    print("SUBMISSION GENERATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate submission file for Kaggle competition")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint (auto-detect if not specified)")
    parser.add_argument("--test_corpus", type=str, default=None,
                        help="Path to test corpus file")
    parser.add_argument("--output", type=str, default="submission.csv",
                        help="Output submission file path")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for predictions")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Use top-k classes instead of threshold")
    parser.add_argument("--min_labels", type=int, default=1,
                        help="Minimum number of labels per sample")
    parser.add_argument("--max_labels", type=int, default=None,
                        help="Maximum number of labels per sample (no limit if not specified)")
    parser.add_argument("--hier_top1", action="store_true",
                        help="Use hierarchical top-1 selection (one class per level)")
    parser.add_argument("--hier_confidence", action="store_true",
                        help="Use confidence-based hierarchical path selection")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Confidence threshold for hierarchical path expansion")
    
    args = parser.parse_args()
    
    generate_submission(
        model_path=args.model_path,
        test_corpus_path=args.test_corpus,
        output_path=args.output,
        threshold=args.threshold,
        top_k=args.top_k,
        min_labels=args.min_labels,
        max_labels=args.max_labels,
        use_hierarchical_top1=args.hier_top1,
        use_hierarchical_confidence=args.hier_confidence,
        confidence_threshold=args.confidence_threshold
    )

