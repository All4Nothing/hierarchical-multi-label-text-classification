"""
Generate submission file with strict hierarchical path constraint
Each document must have a single parent-child path (no siblings)
"""
import os
import csv
import argparse
from typing import List, Set, Dict

import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils.hierarchy import TaxonomyHierarchy
from data.loader import DocumentCorpus, TaxoDataset
from models.classifier import TaxoClassifier, initialize_class_embeddings_with_bert


def load_model(model_path: str, hierarchy: TaxonomyHierarchy, device: str) -> TaxoClassifier:
    """Load trained model from checkpoint"""
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
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model


def predict_classes(
    model: TaxoClassifier,
    test_loader: DataLoader,
    edge_index: torch.Tensor,
    device: str
) -> np.ndarray:
    """Generate predictions for test set"""
    # Register edge_index as buffer
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
            
            predictions = model(input_ids, attention_mask)
            all_predictions.append(predictions.cpu().numpy())
    
    model.set_return_probs(False)
    predictions = np.vstack(all_predictions)
    return predictions


def is_parent_child(class1: int, class2: int, hierarchy: TaxonomyHierarchy) -> bool:
    """Check if class2 is parent or child of class1"""
    # Check if class2 is parent of class1
    parents = hierarchy.get_parents(class1)
    if class2 in parents:
        return True
    
    # Check if class2 is child of class1
    children = hierarchy.get_children(class1)
    if class2 in children:
        return True
    
    return False


def select_hierarchical_path(
    probs: np.ndarray,
    hierarchy: TaxonomyHierarchy,
    threshold: float = 0.1,
    min_labels: int = 2,
    max_labels: int = 3
) -> List[int]:
    """
    Select classes following strict hierarchical path constraint.
    
    Rules:
    1. Start with highest probability class
    2. Add next highest class if it's parent/child of existing path
    3. Continue until min_labels reached or max_labels reached
    4. Result must be a single path: parent->child or parent->child->grandchild
    
    Args:
        probs: Class probabilities
        hierarchy: TaxonomyHierarchy object
        threshold: Minimum probability threshold
        min_labels: Minimum number of labels (default: 2)
        max_labels: Maximum number of labels (default: 3)
    
    Returns:
        List of selected class IDs forming a hierarchical path
    """
    # Get sorted classes by probability
    sorted_indices = np.argsort(probs)[::-1]
    
    # Start with highest probability class
    selected_path = [int(sorted_indices[0])]
    
    # Try to extend the path
    for idx in sorted_indices[1:]:
        class_id = int(idx)
        prob = probs[class_id]
        
        # Stop if below threshold and we have min_labels
        if prob < threshold and len(selected_path) >= min_labels:
            break
        
        # Check if this class extends the current path
        can_add = False
        
        # Option 1: class_id is parent or child of any class in path
        for existing_class in selected_path:
            if is_parent_child(existing_class, class_id, hierarchy):
                can_add = True
                break
        
        if can_add:
            # Verify this maintains a single path (no branching)
            # After adding, check if we still have a valid path
            temp_path = selected_path + [class_id]
            
            if is_valid_path(temp_path, hierarchy):
                selected_path.append(class_id)
                
                # Stop if we reach max_labels
                if len(selected_path) >= max_labels:
                    break
    
    # Ensure we have at least min_labels
    if len(selected_path) < min_labels:
        # Try to extend with any valid parent/child
        for idx in sorted_indices:
            class_id = int(idx)
            if class_id in selected_path:
                continue
            
            # Try to add this class
            for existing_class in selected_path:
                if is_parent_child(existing_class, class_id, hierarchy):
                    temp_path = selected_path + [class_id]
                    if is_valid_path(temp_path, hierarchy):
                        selected_path.append(class_id)
                        break
            
            if len(selected_path) >= min_labels:
                break
    
    # Sort path from root to leaf (by level)
    selected_path = sorted(selected_path, key=lambda c: hierarchy.get_level(c))
    
    return selected_path


def is_valid_path(classes: List[int], hierarchy: TaxonomyHierarchy) -> bool:
    """
    Check if classes form a valid single path (no branching).
    
    Valid paths:
    - [parent, child]
    - [parent, child, grandchild]
    - [grandparent, parent, child]
    
    Invalid paths (branching):
    - [parent, child1, child2] (two children of same parent)
    """
    if len(classes) <= 1:
        return True
    
    # Sort by level
    sorted_classes = sorted(classes, key=lambda c: hierarchy.get_level(c))
    
    # Check each consecutive pair is parent-child
    for i in range(len(sorted_classes) - 1):
        parent = sorted_classes[i]
        child = sorted_classes[i + 1]
        
        # Check if child is actually a child of parent
        children = hierarchy.get_children(parent)
        if child not in children:
            return False
    
    return True


def generate_submission_hierarchy(
    model_path: str = None,
    test_corpus_path: str = None,
    output_path: str = "submission.csv",
    threshold: float = 0.1,
    min_labels: int = 2,
    max_labels: int = 3
):
    """
    Generate submission file with strict hierarchical path constraint
    
    Args:
        model_path: Path to model checkpoint
        test_corpus_path: Path to test corpus file
        output_path: Output submission file path
        threshold: Probability threshold
        min_labels: Minimum number of labels per sample
        max_labels: Maximum number of labels per sample
    """
    print("="*80)
    print(" "*20 + "HIERARCHICAL SUBMISSION GENERATION")
    print("="*80)
    print("\n📋 Method: Strict Hierarchical Path")
    print("   - Each document gets a single parent->child path")
    print("   - No branching (no siblings)")
    print("   - Format: parent->child or parent->child->grandchild")
    print("")

    print(f"min_labels: {min_labels}, max_labels: {max_labels}")
    print(f"threshold: {threshold}")
    
    # Set device
    device = Config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")
    print(f"Using device: {device}")
    
    # Load hierarchy
    print("\nLoading taxonomy hierarchy...")
    hierarchy = TaxonomyHierarchy(Config.HIERARCHY_FILE, Config.CLASSES_FILE)
    
    # Auto-detect model if not specified
    if model_path is None:
        model_path = os.path.join(Config.MODEL_SAVE_DIR, "best_model.pt")
        if not os.path.exists(model_path):
            # Try self-training models
            model_files = [f for f in os.listdir(Config.MODEL_SAVE_DIR) 
                          if f.startswith("self_train_iter_")]
            if model_files:
                iterations = [int(f.replace("self_train_iter_", "").replace(".pt", "")) 
                            for f in model_files]
                max_iter = max(iterations)
                model_path = os.path.join(Config.MODEL_SAVE_DIR, f"self_train_iter_{max_iter}.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Using model: {model_path}")
    
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
    
    # Create dataset and loader
    print("\nCreating test dataset...")
    tokenizer = BertTokenizer.from_pretrained(Config.DOC_ENCODER_MODEL)
    
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
        device=device
    )
    
    # Print prediction statistics
    print(f"\n📊 Prediction Statistics:")
    print(f"   Shape: {predictions.shape}")
    print(f"   Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"   Mean: {predictions.mean():.4f}")
    
    # Sample predictions
    print(f"\n🔍 Sample Predictions (Top-5):")
    for i in range(min(5, len(predictions))):
        top_5 = np.argsort(predictions[i])[-5:][::-1]
        top_5_probs = predictions[i][top_5]
        print(f"   Sample {i}: {top_5.tolist()}, probs: {top_5_probs}")
    
    # Convert predictions to label lists
    print(f"\nConverting predictions to hierarchical paths...")
    print(f"   Threshold: {threshold}")
    print(f"   Min labels: {min_labels}, Max labels: {max_labels}")
    
    all_pids = []
    all_labels = []
    
    # Track statistics
    path_lengths = []
    num_below_threshold = 0
    
    for i, pid in enumerate(tqdm(test_pids, desc="Generating paths")):
        probs = predictions[i]
        
        # Select hierarchical path
        predicted_classes = select_hierarchical_path(
            probs,
            hierarchy=hierarchy,
            threshold=threshold,
            min_labels=min_labels,
            max_labels=max_labels
        )
        
        # Validate path
        if not is_valid_path(predicted_classes, hierarchy):
            print(f"⚠️  Warning: Sample {i} produced invalid path: {predicted_classes}")
        
        path_lengths.append(len(predicted_classes))
        
        # Check if any class is below threshold
        if any(probs[c] < threshold for c in predicted_classes):
            num_below_threshold += 1
        
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
    print(f"\n✅ Submission file saved: {output_path}")
    print(f"\n📈 Path Statistics:")
    print(f"   Total samples: {len(all_pids)}")
    print(f"   Path lengths: min={min(path_lengths)}, max={max(path_lengths)}, avg={np.mean(path_lengths):.2f}")
    print(f"   Samples with classes below threshold: {num_below_threshold} ({100*num_below_threshold/len(all_pids):.1f}%)")
    
    # Analyze path structure
    path_length_dist = {}
    for length in path_lengths:
        path_length_dist[length] = path_length_dist.get(length, 0) + 1
    
    print(f"\n   Path length distribution:")
    for length in sorted(path_length_dist.keys()):
        count = path_length_dist[length]
        pct = 100 * count / len(all_pids)
        print(f"      {length} classes: {count} ({pct:.1f}%)")
    
    # Sample outputs
    print(f"\n🔍 Sample Outputs:")
    for i in range(min(5, len(all_labels))):
        labels = all_labels[i]
        probs_str = ', '.join([f"{predictions[i][c]:.4f}" for c in labels])
        names = [hierarchy.id_to_name.get(c, f"Unknown_{c}") for c in labels]
        print(f"   Sample {i}: {labels} -> {names}")
        print(f"             Probs: [{probs_str}]")
    
    print("\n" + "="*80)
    print("HIERARCHICAL SUBMISSION GENERATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate submission with strict hierarchical path constraint"
    )
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint (auto-detect if not specified)")
    parser.add_argument("--test_corpus", type=str, default=None,
                        help="Path to test corpus file")
    parser.add_argument("--output", type=str, default="submission.csv",
                        help="Output submission file path")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Probability threshold for predictions")
    parser.add_argument("--min_labels", type=int, default=2,
                        help="Minimum number of labels per sample")
    parser.add_argument("--max_labels", type=int, default=3,
                        help="Maximum number of labels per sample")
    
    args = parser.parse_args()
    
    generate_submission_hierarchy(
        model_path=args.model_path,
        test_corpus_path=args.test_corpus,
        output_path=args.output,
        threshold=args.threshold,
        min_labels=args.min_labels,
        max_labels=args.max_labels
    )
