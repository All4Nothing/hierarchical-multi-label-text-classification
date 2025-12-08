"""
Stage 3: TaxoClass Classifier (BERT + GNN)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import os

# Wandb import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class GNNEncoder(nn.Module):
    """Graph Neural Network for encoding class hierarchy"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize 3 encoder
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden dimension
            out_channels: Output dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.batch_norms.append(nn.BatchNorm1d(out_channels))
    
    def forward(self, x, edge_index):
        """
        Forward pass
        
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)
        
        Returns:
            Node embeddings (num_nodes, out_channels)
        """
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GCNConv(nn.Module):
    """Simple GCN convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)
        """
        # Handle DataParallel: ensure edge_index has correct shape (2, num_edges)
        # DataParallel might add extra dimensions, so we need to handle various cases
        original_shape = edge_index.shape
        
        # Case 1: edge_index has extra dimensions like (1, 2, num_edges) or (2, 1, num_edges)
        if edge_index.dim() > 2:
            # Squeeze all dimensions of size 1
            edge_index = edge_index.squeeze()
        
        # Case 2: edge_index is 1D (shouldn't happen, but handle it)
        if edge_index.dim() == 1:
            # Try to reshape to (2, num_edges) - assumes even number of elements
            if edge_index.numel() % 2 == 0:
                edge_index = edge_index.view(2, -1)
            else:
                raise RuntimeError(
                    f"Cannot reshape edge_index from shape {original_shape} to (2, num_edges): "
                    f"number of elements ({edge_index.numel()}) is not even"
                )
        
        # Case 3: edge_index has shape (1, num_edges) instead of (2, num_edges)
        # This can happen if DataParallel incorrectly processes it
        if edge_index.dim() == 2 and edge_index.shape[0] == 1:
            # This is a critical error - we can't reconstruct the missing row
            raise RuntimeError(
                f"edge_index has incorrect shape {original_shape} -> {edge_index.shape}. "
                f"Expected (2, num_edges) but got (1, num_edges). "
                f"This may be caused by DataParallel incorrectly handling the edge_index tensor. "
                f"Consider not using DataParallel or ensuring edge_index is properly replicated to all GPUs."
            )
        
        # Final check: ensure edge_index is 2D with shape (2, num_edges)
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise RuntimeError(
                f"edge_index has incorrect shape {original_shape} -> {edge_index.shape}. "
                f"Expected (2, num_edges) but got {edge_index.shape}"
            )
        
        # Add self-loops
        num_nodes = x.size(0)
        self_loop = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)]).to(edge_index.device)
        edge_index = torch.cat([edge_index, self_loop], dim=1)
        
        # Compute adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes).to(x.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        
        # Degree normalization
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        norm = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        
        # Message passing
        x = self.linear(x)
        x = torch.matmul(norm, x)
        
        return x


class TaxoClassifier(nn.Module):
    """TaxoClass: Document classification with hierarchy-aware GNN"""
    
    def __init__(
        self,
        num_classes: int,
        doc_encoder_name: str = "bert-base-uncased",
        embedding_dim: int = 768,
        gnn_hidden_dim: int = 512,
        gnn_num_layers: int = 3,
        gnn_dropout: float = 0.1,
        freeze_bert: bool = False,
        edge_index: torch.Tensor = None
    ):
        """
        Initialize TaxoClassifier
        
        Args:
            num_classes: Number of classes
            doc_encoder_name: BERT model name
            embedding_dim: Embedding dimension
            gnn_hidden_dim: GNN hidden dimension
            gnn_num_layers: Number of GNN layers
            gnn_dropout: GNN dropout rate
            freeze_bert: Whether to freeze BERT parameters
            edge_index: Edge indices for GNN (optional, can be set later)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Document encoder (BERT)
        self.doc_encoder = BertModel.from_pretrained(doc_encoder_name)
        
        if freeze_bert:
            for param in self.doc_encoder.parameters():
                param.requires_grad = False
        
        # Class encoder (GNN)
        self.class_encoder = GNNEncoder(
            in_channels=embedding_dim,
            hidden_channels=gnn_hidden_dim,
            out_channels=embedding_dim,
            num_layers=gnn_num_layers,
            dropout=gnn_dropout
        )
        
        # Matching matrix B
        self.matching_matrix = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.01)
        
        # Initial class embeddings (learnable)
        # Note: Should be initialized with BERT embeddings using initialize_class_embeddings() method
        # Random initialization is just a placeholder
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.01)
        self.doc_encoder_name = doc_encoder_name  # Store for class embedding initialization
        
        # Register edge_index as buffer to avoid DataParallel splitting it
        # This ensures edge_index is replicated to all GPUs without being split
        if edge_index is not None:
            self.register_buffer('edge_index', edge_index, persistent=False)
        else:
            # Register a dummy buffer that will be set later
            self.register_buffer('edge_index', None, persistent=False)
        
        # Register return_probs as buffer to avoid DataParallel keyword arg issues
        # This allows us to control return_probs without passing it as keyword arg
        self.register_buffer('_return_probs', torch.tensor(False), persistent=False)
    
    def encode_documents(self, input_ids, attention_mask):
        """
        Encode documents using BERT
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            Document embeddings (batch_size, embedding_dim)
        """
        outputs = self.doc_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        doc_emb = outputs.pooler_output  # (batch_size, embedding_dim)
        
        return doc_emb
    
    def encode_classes(self, edge_index=None):
        """
        Encode classes using GNN
        
        Args:
            edge_index: Edge indices (2, num_edges). If None, uses self.edge_index
        
        Returns:
            Class embeddings (num_classes, embedding_dim)
        """
        # Use stored edge_index if not provided (for DataParallel compatibility)
        if edge_index is None:
            edge_index = self.edge_index
        
        if edge_index is None:
            raise ValueError("edge_index must be provided either as argument or registered buffer")
        
        class_emb = self.class_encoder(self.class_embeddings, edge_index)
        return class_emb
    
    def compute_matching_scores(self, doc_emb, class_emb, return_probs=False):
        """
        Compute matching scores between documents and classes
        
        P(y_j=1|D_i) = σ(exp(c_j^T B D_i))
        
        Args:
            doc_emb: Document embeddings (batch_size, embedding_dim)
            class_emb: Class embeddings (num_classes, embedding_dim)
            return_probs: If True, return probabilities (for inference). If False, return logits (for training)
        
        Returns:
            Matching logits or probabilities (batch_size, num_classes)
        """
        # Compute c_j^T B D_i
        # doc_emb: (batch, emb_dim)
        # matching_matrix: (emb_dim, emb_dim)
        # class_emb: (num_classes, emb_dim)
        
        # (batch, emb_dim) @ (emb_dim, emb_dim) = (batch, emb_dim)
        doc_transformed = torch.matmul(doc_emb, self.matching_matrix)
        
        # (batch, emb_dim) @ (num_classes, emb_dim)^T = (batch, num_classes)
        logits = torch.matmul(doc_transformed, class_emb.t())
        
        if return_probs:
            # For inference/evaluation: apply exp and sigmoid as per paper
            # P = σ(exp(c_j^T B D_i))
            # Note: For numerical stability, we use a more stable formulation
            # Instead of exp then sigmoid, we use: sigmoid(logits + log(scale))
            # where scale is a small constant to approximate exp behavior
            # Or simply: sigmoid(logits) for better numerical stability
            # Original paper formula can cause numerical overflow
            # Using sigmoid directly is more stable and produces similar results
            probs = torch.sigmoid(logits)
            return probs
        else:
            # For training: return logits (before exp and sigmoid) for BCEWithLogitsLoss
            # Note: Paper uses exp, but for numerical stability and compatibility with
            # BCEWithLogitsLoss, we use raw logits. The exp can be learned implicitly.
            return logits
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            Prediction logits or probabilities (batch_size, num_classes)
            - Returns probabilities if self._return_probs is True
            - Returns logits if self._return_probs is False (default)
        
        Note:
            edge_index is stored as a model buffer to avoid DataParallel splitting issues.
            return_probs is also stored as a buffer to avoid DataParallel keyword arg issues.
            Use set_return_probs(True/False) to control whether to return probabilities or logits.
        """
        # Encode documents
        doc_emb = self.encode_documents(input_ids, attention_mask)
        
        # Encode classes (always use stored edge_index buffer for DataParallel compatibility)
        class_emb = self.encode_classes(edge_index=None)
        
        # Compute matching scores (use stored return_probs buffer)
        return_probs = bool(self._return_probs.item())
        output = self.compute_matching_scores(doc_emb, class_emb, return_probs=return_probs)
        
        return output
    
    def set_return_probs(self, value: bool):
        """
        Set whether to return probabilities or logits
        
        Args:
            value: If True, return probabilities. If False, return logits.
        """
        self._return_probs.fill_(value)
    
    def initialize_class_embeddings(self, class_names: dict, device: str = "cuda"):
        """
        Initialize class embeddings using BERT
        
        Args:
            class_names: Dictionary mapping class_id to class_name
            device: Device to use for BERT model
        
        Note:
            This method should be called after model initialization to properly
            initialize class embeddings with semantic information from BERT.
        """
        from transformers import BertTokenizer, BertModel
        
        print("Initializing class embeddings with BERT...")
        
        tokenizer = BertTokenizer.from_pretrained(self.doc_encoder_name)
        bert_model = BertModel.from_pretrained(self.doc_encoder_name).to(device)
        bert_model.eval()
        
        embeddings = []
        
        with torch.no_grad():
            for class_id in range(self.num_classes):
                class_name = class_names[class_id].replace('_', ' ')
                
                # Tokenize
                inputs = tokenizer(
                    class_name,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=32
                ).to(device)
                
                # Get embedding
                outputs = bert_model(**inputs)
                emb = outputs.pooler_output.squeeze(0)
                embeddings.append(emb.cpu())
        
        embeddings = torch.stack(embeddings)
        
        # Update class embeddings parameter
        self.class_embeddings.data = embeddings.to(self.class_embeddings.device)
        
        print(f"✅ Initialized class embeddings: {self.class_embeddings.shape}")
        
        # Clean up
        del bert_model
        del tokenizer
        torch.cuda.empty_cache()


class TaxoClassifierTrainer:
    """Trainer for TaxoClassifier"""
    
    def __init__(
        self,
        model: TaxoClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        edge_index: torch.Tensor,
        device: str = "cuda",
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        save_dir: str = "./saved_models",
        use_wandb: bool = False,
        use_mixed_precision: bool = False,
        gradient_accumulation_steps: int = 1,
        use_multi_gpu: bool = False
    ):
        """
        Initialize trainer
        
        Args:
            model: TaxoClassifier model
            train_loader: Training data loader
            val_loader: Validation data loader
            edge_index: Hierarchy edge index
            device: Device to train on
            learning_rate: Learning rate
            num_epochs: Number of epochs
            warmup_steps: Warmup steps for scheduler
            weight_decay: Weight decay
            save_dir: Directory to save models
            use_wandb: Whether to use wandb logging
            use_mixed_precision: Whether to use mixed precision training
            gradient_accumulation_steps: Number of gradient accumulation steps
            use_multi_gpu: Whether to use DataParallel for multi-GPU training
        """
        # Register edge_index as buffer in model before moving to device
        # This ensures DataParallel doesn't split edge_index
        # If edge_index buffer doesn't exist, register it; otherwise update it
        if not hasattr(model, 'edge_index') or model.edge_index is None:
            model.register_buffer('edge_index', edge_index.to(device), persistent=False)
        else:
            # Update existing buffer
            model.edge_index.data = edge_index.to(device)
        
        self.model = model.to(device)
        
        # Wrap model with DataParallel if multiple GPUs are available
        if use_multi_gpu and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            print(f"✅ Using DataParallel on {torch.cuda.device_count()} GPUs")
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.edge_index = edge_index.to(device)  # Keep for backward compatibility
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_mixed_precision = use_mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_multi_gpu = use_multi_gpu
        
        # Mixed precision scaler
        if use_mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')  # Updated API (torch.cuda.amp.GradScaler is deprecated)
            print("✅ Mixed precision training enabled")
        else:
            self.scaler = None
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler (adjust for gradient accumulation)
        total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function: Use BCEWithLogitsLoss for mixed precision compatibility
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # Best validation loss
        self.best_val_loss = float('inf')
        
        # Global step counter for wandb
        self.global_step = 0
        
        os.makedirs(save_dir, exist_ok=True)
        
        if gradient_accumulation_steps > 1:
            print(f"✅ Gradient accumulation enabled: {gradient_accumulation_steps} steps")
            print(f"   Effective batch size: {train_loader.batch_size * gradient_accumulation_steps}")
    
    def compute_loss(self, predictions, labels):
        """
        Compute loss with masking
        
        Args:
            predictions: Predicted logits (batch_size, num_classes)
            labels: Ground truth labels (batch_size, num_classes)
                    1: positive, 0: negative, -1: ignore
        
        Returns:
            Loss value
        """
        # Create mask (ignore labels == -1)
        mask = (labels != -1).float()
        
        # Convert labels to 0/1
        binary_labels = torch.clamp(labels, 0, 1)
        
        # Compute BCEWithLogitsLoss (handles sigmoid internally)
        loss = self.criterion(predictions, binary_labels)
        
        # Apply mask
        loss = loss * mask
        
        # Average over non-masked elements
        loss = loss.sum() / (mask.sum() + 1e-8)
        
        return loss
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        self.optimizer.zero_grad()  # Zero grad at start of epoch
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            # edge_index is stored as model buffer, so we don't need to pass it
            # Don't pass return_probs as keyword arg to avoid DataParallel issues (default is False)
            if self.use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    predictions = self.model(input_ids, attention_mask)
                    loss = self.compute_loss(predictions, labels)
                    loss = loss / self.gradient_accumulation_steps  # Scale loss for accumulation
            else:
                predictions = self.model(input_ids, attention_mask)
                loss = self.compute_loss(predictions, labels)
                loss = loss / self.gradient_accumulation_steps  # Scale loss for accumulation
            
            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update metrics (use unscaled loss for logging)
            unscaled_loss = loss.item() * self.gradient_accumulation_steps
            total_loss += unscaled_loss
            num_batches += 1
            self.global_step += 1
            
            # Log to wandb
            if self.use_wandb and self.global_step % 10 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                wandb.log({
                    "stage3/train_loss": loss.item(),
                    "stage3/learning_rate": current_lr,
                    "stage3/epoch": epoch,
                }, step=self.global_step)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass (return logits for loss computation)
                # edge_index is stored as model buffer, so we don't need to pass it
                # Don't pass return_probs as keyword arg to avoid DataParallel issues (default is False)
                predictions = self.model(input_ids, attention_mask)
                
                # Compute loss
                loss = self.compute_loss(predictions, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        """Train model"""
        print(f"Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "stage3/epoch_train_loss": train_loss,
                    "stage3/epoch_val_loss": val_loss,
                    "stage3/epoch": epoch + 1,
                }, step=self.global_step)
                
                # Log best model update
                if val_loss < self.best_val_loss:
                    wandb.log({
                        "stage3/best_val_loss": val_loss,
                        "stage3/best_epoch": epoch + 1,
                    }, step=self.global_step)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(f"best_model.pt")
                print(f"Saved best model with validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(f"checkpoint_epoch_{epoch+1}.pt")
        
        print("Training complete!")
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        save_path = os.path.join(self.save_dir, filename)
        
        # Handle DataParallel: save underlying model state_dict
        model_state_dict = self.model.state_dict()
        if isinstance(self.model, torch.nn.DataParallel):
            # DataParallel wraps the model in a 'module' attribute
            model_state_dict = self.model.module.state_dict()
        
        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }, save_path)
    
    def load_model(self, filename: str):
        """Load model checkpoint"""
        load_path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(load_path)
        
        # Handle DataParallel: load into underlying model
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded model from {load_path}")


def initialize_class_embeddings_with_bert(
    class_names: Dict[int, str],
    bert_model_name: str = "bert-base-uncased",
    device: str = "cuda"
) -> torch.Tensor:
    """
    Initialize class embeddings using BERT
    
    Args:
        class_names: Dictionary mapping class_id to class_name
        bert_model_name: BERT model name
        device: Device
    
    Returns:
        Class embeddings tensor (num_classes, embedding_dim)
    """
    print("Initializing class embeddings with BERT...")
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    model = BertModel.from_pretrained(bert_model_name).to(device)
    model.eval()
    
    embeddings = []
    
    with torch.no_grad():
        for class_id in range(len(class_names)):
            class_name = class_names[class_id].replace('_', ' ')
            
            # Tokenize
            inputs = tokenizer(
                class_name,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=32
            ).to(device)
            
            # Get embedding
            outputs = model(**inputs)
            emb = outputs.pooler_output.squeeze(0)
            embeddings.append(emb.cpu())
    
    embeddings = torch.stack(embeddings)
    
    print(f"Initialized class embeddings: {embeddings.shape}")
    
    return embeddings

