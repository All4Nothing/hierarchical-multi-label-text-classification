"""
Evaluation Metrics for Hierarchical Multi-label Classification
"""
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
import torch


class HierarchicalMetrics:
    """Metrics for hierarchical multi-label classification"""
    
    def __init__(self, hierarchy):
        """
        Initialize metrics calculator
        
        Args:
            hierarchy: TaxonomyHierarchy object
        """
        self.hierarchy = hierarchy
    
    def compute_f1_scores(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute micro and macro F1 scores
        
        Args:
            predictions: Predicted probabilities (num_samples, num_classes)
            ground_truth: Ground truth labels (num_samples, num_classes)
            threshold: Threshold for binary prediction
        
        Returns:
            Dictionary of F1 scores
        """
        # Convert predictions to binary
        binary_preds = (predictions >= threshold).astype(int)
        
        # Compute F1 scores
        micro_f1 = f1_score(ground_truth, binary_preds, average='micro', zero_division=0)
        macro_f1 = f1_score(ground_truth, binary_preds, average='macro', zero_division=0)
        
        return {
            'micro_f1': micro_f1,
            'macro_f1': macro_f1
        }
    
    def compute_precision_recall(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute precision and recall scores
        
        Args:
            predictions: Predicted probabilities
            ground_truth: Ground truth labels
            threshold: Threshold for binary prediction
        
        Returns:
            Dictionary of precision and recall scores
        """
        binary_preds = (predictions >= threshold).astype(int)
        
        micro_precision = precision_score(ground_truth, binary_preds, average='micro', zero_division=0)
        macro_precision = precision_score(ground_truth, binary_preds, average='macro', zero_division=0)
        micro_recall = recall_score(ground_truth, binary_preds, average='micro', zero_division=0)
        macro_recall = recall_score(ground_truth, binary_preds, average='macro', zero_division=0)
        
        return {
            'micro_precision': micro_precision,
            'macro_precision': macro_precision,
            'micro_recall': micro_recall,
            'macro_recall': macro_recall
        }
    
    def compute_hierarchical_precision_recall(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        k: int = 5
    ) -> Dict[str, float]:
        """
        Compute hierarchical precision and recall at top-k
        
        Args:
            predictions: Predicted probabilities
            ground_truth: Ground truth labels
            k: Number of top predictions to consider
        
        Returns:
            Dictionary of hierarchical metrics
        """
        num_samples = predictions.shape[0]
        
        h_precision = 0.0
        h_recall = 0.0
        
        for i in range(num_samples):
            # Get top-k predictions
            top_k_indices = np.argsort(predictions[i])[-k:][::-1]
            
            # Get predicted classes and their ancestors
            predicted_set = set(top_k_indices)
            for class_id in top_k_indices:
                predicted_set.update(self.hierarchy.get_ancestors(class_id))
            
            # Get true classes and their ancestors
            true_indices = np.where(ground_truth[i] == 1)[0]
            true_set = set(true_indices)
            for class_id in true_indices:
                true_set.update(self.hierarchy.get_ancestors(class_id))
            
            # Compute precision and recall
            if len(predicted_set) > 0:
                precision = len(predicted_set & true_set) / len(predicted_set)
                h_precision += precision
            
            if len(true_set) > 0:
                recall = len(predicted_set & true_set) / len(true_set)
                h_recall += recall
        
        h_precision /= num_samples
        h_recall /= num_samples
        
        h_f1 = 2 * h_precision * h_recall / (h_precision + h_recall + 1e-10)
        
        return {
            f'hierarchical_precision@{k}': h_precision,
            f'hierarchical_recall@{k}': h_recall,
            f'hierarchical_f1@{k}': h_f1
        }
    
    def compute_ndcg(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        k: int = 10
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain (nDCG@k)
        
        Args:
            predictions: Predicted probabilities
            ground_truth: Ground truth labels
            k: Number of top predictions
        
        Returns:
            nDCG@k score
        """
        num_samples = predictions.shape[0]
        ndcg_sum = 0.0
        
        for i in range(num_samples):
            # Get top-k predictions
            top_k_indices = np.argsort(predictions[i])[-k:][::-1]
            
            # Compute DCG
            dcg = 0.0
            for j, class_id in enumerate(top_k_indices):
                relevance = ground_truth[i, class_id]
                dcg += relevance / np.log2(j + 2)  # +2 because positions start from 1
            
            # Compute ideal DCG
            ideal_relevances = np.sort(ground_truth[i])[-k:][::-1]
            idcg = 0.0
            for j, relevance in enumerate(ideal_relevances):
                idcg += relevance / np.log2(j + 2)
            
            # Compute nDCG
            if idcg > 0:
                ndcg = dcg / idcg
                ndcg_sum += ndcg
        
        return ndcg_sum / num_samples
    
    def compute_hamming_loss(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """
        Compute Hamming Loss
        
        Args:
            predictions: Predicted probabilities
            ground_truth: Ground truth labels
            threshold: Threshold for binary prediction
        
        Returns:
            Hamming loss
        """
        binary_preds = (predictions >= threshold).astype(int)
        return hamming_loss(ground_truth, binary_preds)
    
    def compute_all_metrics(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float = 0.5,
        top_k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics
        
        Args:
            predictions: Predicted probabilities
            ground_truth: Ground truth labels
            threshold: Threshold for binary prediction
            top_k_values: List of k values for top-k metrics
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # F1 scores
        metrics.update(self.compute_f1_scores(predictions, ground_truth, threshold))
        
        # Precision and recall
        metrics.update(self.compute_precision_recall(predictions, ground_truth, threshold))
        
        # Hamming loss
        metrics['hamming_loss'] = self.compute_hamming_loss(predictions, ground_truth, threshold)
        
        # Hierarchical metrics at different k values
        for k in top_k_values:
            metrics.update(self.compute_hierarchical_precision_recall(predictions, ground_truth, k))
            metrics[f'ndcg@{k}'] = self.compute_ndcg(predictions, ground_truth, k)
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in a formatted way"""
        print("\n" + "="*60)
        print("Evaluation Metrics")
        print("="*60)
        
        # F1 scores
        print("\nF1 Scores:")
        print(f"  Micro-F1: {metrics.get('micro_f1', 0):.4f}")
        print(f"  Macro-F1: {metrics.get('macro_f1', 0):.4f}")
        
        # Precision and Recall
        print("\nPrecision & Recall:")
        print(f"  Micro-Precision: {metrics.get('micro_precision', 0):.4f}")
        print(f"  Macro-Precision: {metrics.get('macro_precision', 0):.4f}")
        print(f"  Micro-Recall: {metrics.get('micro_recall', 0):.4f}")
        print(f"  Macro-Recall: {metrics.get('macro_recall', 0):.4f}")
        
        # Hamming Loss
        print(f"\nHamming Loss: {metrics.get('hamming_loss', 0):.4f}")
        
        # Hierarchical metrics
        print("\nHierarchical Metrics:")
        for k in [1, 3, 5, 10]:
            if f'hierarchical_precision@{k}' in metrics:
                print(f"\n  Top-{k}:")
                print(f"    H-Precision: {metrics[f'hierarchical_precision@{k}']:.4f}")
                print(f"    H-Recall: {metrics[f'hierarchical_recall@{k}']:.4f}")
                print(f"    H-F1: {metrics[f'hierarchical_f1@{k}']:.4f}")
                print(f"    nDCG: {metrics[f'ndcg@{k}']:.4f}")
        
        print("="*60 + "\n")


def evaluate_model(
    model,
    test_loader,
    edge_index: torch.Tensor,
    ground_truth: np.ndarray,
    hierarchy,
    device: str = "cuda",
    threshold: float = 0.5,
    top_k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate model on test data
    
    Args:
        model: TaxoClassifier model
        test_loader: Test data loader
        edge_index: Hierarchy edge index
        ground_truth: Ground truth labels
        hierarchy: TaxonomyHierarchy object
        device: Device
        threshold: Prediction threshold
        top_k_values: List of k values
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    all_predictions = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            predictions = model(input_ids, attention_mask, edge_index, return_probs=True)
            all_predictions.append(predictions.cpu().numpy())
    
    predictions = np.vstack(all_predictions)
    
    print("Computing metrics...")
    metrics_calculator = HierarchicalMetrics(hierarchy)
    metrics = metrics_calculator.compute_all_metrics(
        predictions,
        ground_truth,
        threshold=threshold,
        top_k_values=top_k_values
    )
    
    metrics_calculator.print_metrics(metrics)
    
    return metrics


def predict_top_k_classes(
    model,
    documents: List[str],
    tokenizer,
    edge_index: torch.Tensor,
    hierarchy,
    device: str = "cuda",
    k: int = 5,
    batch_size: int = 32
) -> List[List[Tuple[int, str, float]]]:
    """
    Predict top-k classes for documents
    
    Args:
        model: TaxoClassifier model
        documents: List of document texts
        tokenizer: Tokenizer
        edge_index: Hierarchy edge index
        hierarchy: TaxonomyHierarchy object
        device: Device
        k: Number of top classes
        batch_size: Batch size
    
    Returns:
        List of [(class_id, class_name, score)] for each document
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            
            # Tokenize
            encodings = tokenizer(
                batch_docs,
                max_length=256,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            # Predict (return probabilities for inference)
            predictions = model(input_ids, attention_mask, edge_index, return_probs=True)
            predictions = predictions.cpu().numpy()
            
            # Get top-k for each document
            for j in range(len(batch_docs)):
                top_k_indices = np.argsort(predictions[j])[-k:][::-1]
                top_k_results = []
                
                for class_id in top_k_indices:
                    class_name = hierarchy.id_to_name.get(class_id, "Unknown")
                    score = predictions[j, class_id]
                    top_k_results.append((class_id, class_name, score))
                
                results.append(top_k_results)
    
    return results

