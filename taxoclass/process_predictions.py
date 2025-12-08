"""
Process saved predictions and generate submission file
Loads predictions from file and converts them to submission format
"""
import os
import csv
from typing import List

import numpy as np
from tqdm import tqdm

from config import Config
from utils.hierarchy import TaxonomyHierarchy

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


def process_predictions(
    predictions_path: str = "predictions.npy",
    test_pids_path: str = "test_pids.txt",
    output_path: str = "submission.csv",
    threshold: float = 0.5,
    min_labels: int = 2,
    max_labels: int = 3,  # Kaggle rule: at least 2 and at most 3 labels
    use_hierarchical_top1: bool = False,
    use_hierarchical_confidence: bool = False,
    confidence_threshold: float = 0.5
):
    """
    Process saved predictions and generate submission file
    
    Args:
        predictions_path: Path to saved predictions numpy file
        test_pids_path: Path to saved test PIDs text file
        output_path: Output submission file path
        threshold: Probability threshold for predictions
        min_labels: Minimum number of labels per sample
        max_labels: Maximum number of labels per sample
        use_hierarchical_top1: Use hierarchical top-1 selection
        use_hierarchical_confidence: Use confidence-based hierarchical path selection
        confidence_threshold: Confidence threshold for hierarchical path expansion
    """
    print("="*80)
    print(" "*25 + "PROCESS PREDICTIONS")
    print("="*80)
    
    # Load predictions
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    
    print(f"\nLoading predictions from {predictions_path}...")
    predictions = np.load(predictions_path)
    print(f"✅ Predictions loaded: shape={predictions.shape}")
    
    # Load test PIDs
    if not os.path.exists(test_pids_path):
        raise FileNotFoundError(f"Test PIDs file not found: {test_pids_path}")
    
    print(f"\nLoading test PIDs from {test_pids_path}...")
    test_pids = []
    with open(test_pids_path, 'r', encoding='utf-8') as f:
        for line in f:
            pid = line.strip()
            if pid:
                test_pids.append(pid)
    print(f"✅ Test PIDs loaded: {len(test_pids)} samples")
    
    # Verify consistency
    if len(test_pids) != len(predictions):
        raise ValueError(
            f"Mismatch: {len(test_pids)} PIDs but {len(predictions)} predictions"
        )
    
    # Load hierarchy
    print("\nLoading taxonomy hierarchy...")
    hierarchy = TaxonomyHierarchy(Config.HIERARCHY_FILE, Config.CLASSES_FILE)
    
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
    
    # Prepare level/structure caches for hierarchical selection & consistency
    global LEVEL_NODES_CACHE
    if use_hierarchical_top1 or use_hierarchical_confidence:
        LEVEL_NODES_CACHE = {}
        for node, level in hierarchy.levels.items():
            LEVEL_NODES_CACHE.setdefault(level, []).append(node)

    leaves_set = set(hierarchy.get_leaves())
    
    # Convert predictions to label lists
    print("\nConverting predictions to submission format...")
    all_pids = []
    all_labels = []
    
    for i, pid in enumerate(tqdm(test_pids, desc="Formatting predictions")):
        probs = predictions[i]
        
        # Apply base selection strategy (probability-based)
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
        else:
            # Threshold-based
            predicted_classes = np.where(probs >= threshold)[0].tolist()
        
        # ------------------------------------------------------------------
        # Enforce taxonomy consistency + 2~3 label constraint
        # ------------------------------------------------------------------
        # 1) If any class is predicted, close it upward to its ancestors
        if predicted_classes:
            closure = set()
            for cid in predicted_classes:
                closure.add(int(cid))
                for anc in hierarchy.get_ancestors(int(cid)):
                    closure.add(int(anc))
        else:
            # Fallback: start from global best class
            best = int(np.argmax(probs))
            closure = set([best, *list(hierarchy.get_ancestors(best))])

        closure = list(closure)

        # 2) Pick a leaf-centric path (leaf + its parents) as main prediction
        leaf_candidates = [c for c in closure if c in leaves_set]
        if leaf_candidates:
            best_leaf = max(leaf_candidates, key=lambda c: probs[c])
        else:
            best_leaf = max(closure, key=lambda c: probs[c])

        path_nodes = list(hierarchy.get_ancestors(best_leaf)) + [best_leaf]
        # sort path from root -> leaf
        path_nodes = sorted(path_nodes, key=lambda c: hierarchy.get_level(c))

        # 3) Enforce 2~3 labels using this path
        if len(path_nodes) >= max_labels:
            # take deepest max_labels nodes (closer to leaf)
            selected = path_nodes[-max_labels:]
        elif len(path_nodes) == 1:
            selected = path_nodes[:]
            # need at least 2 labels: add next best class not in path
            extra_candidates = [c for c in closure if c not in selected]
            if extra_candidates:
                extra = max(extra_candidates, key=lambda c: probs[c])
                selected.append(extra)
        else:
            # len(path_nodes) == 2 and max_labels == 3: keep as is (already >= min_labels)
            selected = path_nodes[:]

        # If still fewer than min_labels, pad with global top classes
        if len(selected) < min_labels:
            for idx in np.argsort(probs)[::-1]:
                idx = int(idx)
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= min_labels:
                    break

        # If more than max_labels (edge case), keep highest-prob classes
        if max_labels is not None and len(selected) > max_labels:
            selected = sorted(selected, key=lambda c: probs[c], reverse=True)[:max_labels]

        predicted_classes = selected
        
        # Sort class IDs for stable output
        predicted_classes = sorted(set(int(c) for c in predicted_classes))
        
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
    
    parser = argparse.ArgumentParser(description="Process saved predictions and generate submission file")
    parser.add_argument("--predictions", type=str, default="predictions.npy",
                        help="Path to saved predictions numpy file")
    parser.add_argument("--test_pids", type=str, default="test_pids.txt",
                        help="Path to saved test PIDs text file")
    parser.add_argument("--output", type=str, default="submission.csv",
                        help="Output submission file path")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for predictions")
    parser.add_argument("--min_labels", type=int, default=2,
                        help="Minimum number of labels per sample")
    parser.add_argument("--max_labels", type=int, default=3,
                        help="Maximum number of labels per sample")
    parser.add_argument("--hier_confidence", action="store_true",
                        help="Use confidence-based hierarchical path selection")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Confidence threshold for hierarchical path expansion")
    
    args = parser.parse_args()
    
    process_predictions(
        predictions_path=args.predictions,
        test_pids_path=args.test_pids,
        output_path=args.output,
        threshold=args.threshold,
        min_labels=args.min_labels,
        max_labels=args.max_labels,
        use_hierarchical_confidence=args.hier_confidence,
        confidence_threshold=args.confidence_threshold
    )
