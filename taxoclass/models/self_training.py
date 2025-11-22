"""
Stage 4: Multi-label Self-Training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List
from tqdm import tqdm
import os


class SelfTrainer:
    """Self-training for TaxoClassifier"""
    
    def __init__(
        self,
        model,
        unlabeled_loader: DataLoader,
        edge_index: torch.Tensor,
        device: str = "cuda",
        num_iterations: int = 5,
        num_epochs_per_iter: int = 3,
        temperature: float = 2.0,
        threshold: float = 0.5,
        learning_rate: float = 1e-5,
        save_dir: str = "./saved_models"
    ):
        """
        Initialize self-trainer
        
        Args:
            model: TaxoClassifier model
            unlabeled_loader: DataLoader for unlabeled data
            edge_index: Hierarchy edge index
            device: Device to train on
            num_iterations: Number of self-training iterations
            num_epochs_per_iter: Epochs per iteration
            temperature: Temperature for target distribution
            threshold: Threshold for prediction filtering
            learning_rate: Learning rate for self-training
            save_dir: Directory to save models
        """
        self.model = model.to(device)
        self.unlabeled_loader = unlabeled_loader
        self.edge_index = edge_index.to(device)
        self.device = device
        self.num_iterations = num_iterations
        self.num_epochs_per_iter = num_epochs_per_iter
        self.temperature = temperature
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
    
    def compute_target_distribution(
        self,
        predictions: torch.Tensor,
        temperature: float = None,
        threshold: float = None
    ) -> torch.Tensor:
        """
        Compute target distribution Q from predictions P
        
        Q_ij = P_ij^(1/T) / Σ_k P_ik^(1/T)  if P_ij > threshold
             = 0                           otherwise
        
        Args:
            predictions: Model predictions (num_samples, num_classes)
            temperature: Temperature parameter (default: self.temperature)
            threshold: Threshold parameter (default: self.threshold)
        
        Returns:
            Target distribution Q (num_samples, num_classes)
        """
        if temperature is None:
            temperature = self.temperature
        if threshold is None:
            threshold = self.threshold
        
        # Temperature scaling
        Q = torch.pow(predictions, 1.0 / temperature)
        
        # Apply threshold
        Q[predictions < threshold] = 0.0
        
        # Normalize (per sample)
        Q_sum = Q.sum(dim=1, keepdim=True)
        Q = Q / (Q_sum + 1e-10)
        
        # Handle cases where all predictions are below threshold
        zero_rows = (Q_sum.squeeze() == 0)
        if zero_rows.any():
            # Keep original predictions for these samples
            Q[zero_rows] = predictions[zero_rows]
        
        return Q
    
    def predict_all(self) -> torch.Tensor:
        """
        Get predictions for all unlabeled data
        
        Returns:
            Predictions tensor (num_samples, num_classes)
        """
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(self.unlabeled_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                predictions = self.model(input_ids, attention_mask, self.edge_index)
                all_predictions.append(predictions.cpu())
        
        all_predictions = torch.cat(all_predictions, dim=0)
        return all_predictions
    
    def kl_divergence_loss(
        self,
        predictions: torch.Tensor,
        target_distribution: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence: D_KL(Q || P)
        
        Args:
            predictions: Model predictions P
            target_distribution: Target distribution Q
        
        Returns:
            KL divergence loss
        """
        # KL(Q || P) = Σ Q * log(Q / P)
        #            = Σ Q * (log(Q) - log(P))
        
        # Add small epsilon for numerical stability
        eps = 1e-10
        log_predictions = torch.log(predictions + eps)
        log_target = torch.log(target_distribution + eps)
        
        # Only compute loss for non-zero target probabilities
        mask = (target_distribution > 0).float()
        
        kl_loss = target_distribution * (log_target - log_predictions)
        kl_loss = (kl_loss * mask).sum(dim=1).mean()
        
        return kl_loss
    
    def train_iteration(
        self,
        iteration: int,
        target_distribution: torch.Tensor
    ):
        """
        Train for one self-training iteration
        
        Args:
            iteration: Iteration number
            target_distribution: Target distribution Q
        """
        self.model.train()
        
        # Create optimizer for this iteration
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        print(f"\nSelf-Training Iteration {iteration + 1}/{self.num_iterations}")
        
        for epoch in range(self.num_epochs_per_iter):
            total_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(
                self.unlabeled_loader,
                desc=f"  Epoch {epoch+1}/{self.num_epochs_per_iter}"
            )
            
            batch_start_idx = 0
            
            for batch in pbar:
                batch_size = batch['input_ids'].size(0)
                batch_end_idx = batch_start_idx + batch_size
                
                # Get target distribution for this batch
                batch_targets = target_distribution[batch_start_idx:batch_end_idx].to(self.device)
                
                # Move inputs to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                predictions = self.model(input_ids, attention_mask, self.edge_index)
                
                # Compute KL divergence loss
                loss = self.kl_divergence_loss(predictions, batch_targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                batch_start_idx = batch_end_idx
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
    
    def self_train(self):
        """
        Run full self-training process
        """
        print("="*60)
        print("Starting Self-Training")
        print("="*60)
        
        for iteration in range(self.num_iterations):
            # Step 1: Get predictions on all unlabeled data
            print(f"\nIteration {iteration + 1}: Generating predictions...")
            predictions = self.predict_all()
            
            # Step 2: Compute target distribution
            print("Computing target distribution...")
            target_distribution = self.compute_target_distribution(predictions)
            
            # Print statistics
            num_confident = (predictions.max(dim=1)[0] > self.threshold).sum().item()
            print(f"Confident predictions: {num_confident}/{len(predictions)} ({100*num_confident/len(predictions):.2f}%)")
            print(f"Avg max prediction: {predictions.max(dim=1)[0].mean():.4f}")
            print(f"Avg target entropy: {-(target_distribution * torch.log(target_distribution + 1e-10)).sum(dim=1).mean():.4f}")
            
            # Step 3: Train model with target distribution
            self.train_iteration(iteration, target_distribution)
            
            # Save checkpoint
            self.save_model(f"self_train_iter_{iteration+1}.pt")
        
        print("\n" + "="*60)
        print("Self-Training Complete!")
        print("="*60)
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        save_path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, save_path)
        print(f"Saved model to {save_path}")
    
    def load_model(self, filename: str):
        """Load model checkpoint"""
        load_path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {load_path}")


class AdaptiveThresholdSelfTrainer(SelfTrainer):
    """Self-trainer with adaptive threshold"""
    
    def __init__(self, *args, initial_threshold: float = 0.7, threshold_decay: float = 0.9, **kwargs):
        """
        Initialize adaptive threshold self-trainer
        
        Args:
            initial_threshold: Initial threshold value
            threshold_decay: Decay factor for threshold per iteration
        """
        super().__init__(*args, **kwargs)
        self.initial_threshold = initial_threshold
        self.threshold_decay = threshold_decay
        self.current_threshold = initial_threshold
    
    def self_train(self):
        """Run self-training with adaptive threshold"""
        print("="*60)
        print("Starting Adaptive Threshold Self-Training")
        print("="*60)
        
        self.current_threshold = self.initial_threshold
        
        for iteration in range(self.num_iterations):
            print(f"\nIteration {iteration + 1}: Threshold = {self.current_threshold:.4f}")
            
            # Step 1: Get predictions
            print("Generating predictions...")
            predictions = self.predict_all()
            
            # Step 2: Compute target distribution with current threshold
            print("Computing target distribution...")
            target_distribution = self.compute_target_distribution(
                predictions,
                threshold=self.current_threshold
            )
            
            # Print statistics
            num_confident = (predictions.max(dim=1)[0] > self.current_threshold).sum().item()
            print(f"Confident predictions: {num_confident}/{len(predictions)} ({100*num_confident/len(predictions):.2f}%)")
            
            # Step 3: Train
            self.train_iteration(iteration, target_distribution)
            
            # Save checkpoint
            self.save_model(f"adaptive_self_train_iter_{iteration+1}.pt")
            
            # Decay threshold
            self.current_threshold *= self.threshold_decay
            self.current_threshold = max(self.current_threshold, 0.3)  # Minimum threshold
        
        print("\n" + "="*60)
        print("Adaptive Self-Training Complete!")
        print("="*60)


def create_unlabeled_dataset(
    documents: List[str],
    tokenizer,
    max_length: int = 256,
    batch_size: int = 32
) -> DataLoader:
    """
    Create DataLoader for unlabeled data
    
    Args:
        documents: List of document texts
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size
    
    Returns:
        DataLoader for unlabeled data
    """
    print(f"Creating unlabeled dataset with {len(documents)} documents...")
    
    input_ids_list = []
    attention_mask_list = []
    
    for doc in tqdm(documents, desc="Tokenizing"):
        encoding = tokenizer(
            doc,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids_list.append(encoding['input_ids'].squeeze(0))
        attention_mask_list.append(encoding['attention_mask'].squeeze(0))
    
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    
    # Create dummy labels tensor (not used in self-training)
    dummy_labels = torch.zeros(len(documents), 1)
    
    dataset = TensorDataset(input_ids, attention_mask, dummy_labels)
    
    # Create custom collate function
    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return loader

