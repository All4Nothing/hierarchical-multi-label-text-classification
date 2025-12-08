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



def select_pure_threshold(
    probs: np.ndarray,
    hierarchy,
    threshold: float = 0.5,
    min_labels: int = 2,
    max_labels: int = 3
) -> List[int]:
    """
    Pure threshold + ancestor closure method (paper method)
    
    This method follows the TaxoClass paper more closely:
    1. Select all classes above threshold
    2. Add ancestors for hierarchical consistency
    3. Select top-K by probability
    
    Args:
        probs: Class probabilities
        hierarchy: TaxonomyHierarchy object
        threshold: Probability threshold
        min_labels: Minimum number of labels
        max_labels: Maximum number of labels
    
    Returns:
        Selected class indices
    """
    # Step 1: Threshold-based selection
    predicted = np.where(probs >= threshold)[0].tolist()
    
    if not predicted:
        # Fallback: select top class
        predicted = [int(np.argmax(probs))]
    
    # Step 2: Add ancestors for hierarchical consistency
    closure = set()
    for cls_id in predicted:
        closure.add(cls_id)
        # Add ancestors
        ancestors = hierarchy.get_ancestors(cls_id)
        closure.update(ancestors)
    
    # Step 3: Select top-K by probability
    closure_list = list(closure)
    closure_sorted = sorted(closure_list, key=lambda c: probs[c], reverse=True)
    
    # Apply max_labels constraint
    if max_labels is not None:
        selected = closure_sorted[:max_labels]
    else:
        selected = closure_sorted
    
    # Ensure min_labels
    if len(selected) < min_labels:
        # Add more classes from original predictions
        for idx in np.argsort(probs)[::-1]:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
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
        state_dict = checkpoint['model_state_dict']
        print("   Loaded from 'model_state_dict'")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("   Loaded from 'state_dict'")
    else:
        # Self-training models might only have state_dict at root
        state_dict = checkpoint
        print("   Loaded from root level (self-training format)")
    
    # Check if state_dict has 'module.' prefix (saved from DataParallel)
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    
    if has_module_prefix:
        print("   Detected DataParallel model, removing 'module.' prefix...")
        # Remove 'module.' prefix from all keys
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load state dict (use strict=False to ignore edge_index buffer if present)
    model.load_state_dict(state_dict, strict=False)
    print("   ✅ State dict loaded successfully")
    
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
) -> np.ndarray:
    """
    Generate predictions for test set
    
    Args:
        model: Trained TaxoClassifier model
        test_loader: Test data loader
        edge_index: Hierarchy edge index
        device: Device
        threshold: Probability threshold for binary classification
        
    Returns:
        Probability predictions array (num_samples, num_classes)
    """
    # Register edge_index as buffer in model (DataParallel-safe)
    if not hasattr(model, 'edge_index') or model.edge_index is None:
        model.register_buffer('edge_index', edge_index.to(device), persistent=False)
    else:
        model.edge_index.data = edge_index.to(device)
    
    # Set return_probs to True for inference
    model.set_return_probs(True)
    
    all_predictions = []
    
    print("\nGenerating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get probabilities (edge_index is stored as model buffer)
            predictions = model(input_ids, attention_mask)
            all_predictions.append(predictions.cpu().numpy())
    
    # Reset return_probs to False
    model.set_return_probs(False)
    
    predictions = np.vstack(all_predictions)
    return predictions


def generate_submission(
    model_path: str = None,
    test_corpus_path: str = None,
    output_path: str = "submission.csv",
    threshold: float = 0.5,
    min_labels: int = 2,
    max_labels: int = 3,  # Kaggle rule: at least 2 and at most 3 labels
):
    """
    Generate submission file for Kaggle competition
    
    Args:
        model_path: Path to model checkpoint (if None, auto-detect best model)
        test_corpus_path: Path to test corpus file
        output_path: Output submission file path
        threshold: Probability threshold for predictions
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
    edge_index = torch.LongTensor(
        hierarchy.get_edge_index(bidirectional=Config.GNN_BIDIRECTIONAL_EDGES)
    ).to(device)
    
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
    
    """for debugging"""
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
    )
    
    # Save predictions
    predictions_path = "predictions.npy"
    print(f"\n💾 Saving predictions to {predictions_path}...")
    np.save(predictions_path, predictions)
    print(f"✅ Predictions saved: shape={predictions.shape}")
    
    # Also save test_pids for later use
    test_pids_path = "test_pids.txt"
    with open(test_pids_path, 'w', encoding='utf-8') as f:
        for pid in test_pids:
            f.write(f"{pid}\n")
    print(f"✅ Test PIDs saved to {test_pids_path}")
    
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
    
    leaves_set = set(hierarchy.get_leaves())
    
    # Convert predictions to label lists
    print("\nConverting predictions to submission format...")
    

    print("📊 Using: Pure Threshold + Ancestor Closure")

    
    all_pids = []
    all_labels = []
    
    for i, pid in enumerate(tqdm(test_pids, desc="Formatting predictions")):
        probs = predictions[i]
        
        predicted_classes = select_pure_threshold(
            probs,
            hierarchy=hierarchy,
            threshold=threshold,
            min_labels=min_labels,
            max_labels=max_labels
        )

        all_pids.append(pid)
        all_labels.append(sorted(predicted_classes))
    
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
    parser.add_argument("--min_labels", type=int, default=2,
                        help="Minimum number of labels per sample")
    parser.add_argument("--max_labels", type=int, default=3,
                        help="Maximum number of labels per sample (no limit if not specified)")

    args = parser.parse_args()
    
    generate_submission(
        model_path=args.model_path,
        test_corpus_path=args.test_corpus,
        output_path=args.output,
        threshold=args.threshold,
        min_labels=args.min_labels,
        max_labels=args.max_labels,
    )

