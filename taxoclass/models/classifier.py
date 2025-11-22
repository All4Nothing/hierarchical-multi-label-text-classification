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
        Initialize GNN encoder
        
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
        freeze_bert: bool = False
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
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.01)
    
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
    
    def encode_classes(self, edge_index):
        """
        Encode classes using GNN
        
        Args:
            edge_index: Edge indices (2, num_edges)
        
        Returns:
            Class embeddings (num_classes, embedding_dim)
        """
        class_emb = self.class_encoder(self.class_embeddings, edge_index)
        return class_emb
    
    def compute_matching_scores(self, doc_emb, class_emb):
        """
        Compute matching scores between documents and classes
        
        P(y_j=1|D_i) = σ(exp(c_j^T B D_i))
        
        Args:
            doc_emb: Document embeddings (batch_size, embedding_dim)
            class_emb: Class embeddings (num_classes, embedding_dim)
        
        Returns:
            Matching probabilities (batch_size, num_classes)
        """
        # Compute c_j^T B D_i
        # doc_emb: (batch, emb_dim)
        # matching_matrix: (emb_dim, emb_dim)
        # class_emb: (num_classes, emb_dim)
        
        # (batch, emb_dim) @ (emb_dim, emb_dim) = (batch, emb_dim)
        doc_transformed = torch.matmul(doc_emb, self.matching_matrix)
        
        # (batch, emb_dim) @ (num_classes, emb_dim)^T = (batch, num_classes)
        scores = torch.matmul(doc_transformed, class_emb.t())
        
        # Apply exp and sigmoid
        scores = torch.exp(scores)
        probs = torch.sigmoid(scores)
        
        return probs
    
    def forward(self, input_ids, attention_mask, edge_index):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            edge_index: Edge indices (2, num_edges)
        
        Returns:
            Prediction probabilities (batch_size, num_classes)
        """
        # Encode documents
        doc_emb = self.encode_documents(input_ids, attention_mask)
        
        # Encode classes
        class_emb = self.encode_classes(edge_index)
        
        # Compute matching scores
        probs = self.compute_matching_scores(doc_emb, class_emb)
        
        return probs


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
        use_wandb: bool = False
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
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.edge_index = edge_index.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        self.criterion = nn.BCELoss(reduction='none')
        
        # Best validation loss
        self.best_val_loss = float('inf')
        
        # Global step counter for wandb
        self.global_step = 0
        
        os.makedirs(save_dir, exist_ok=True)
    
    def compute_loss(self, predictions, labels):
        """
        Compute loss with masking
        
        Args:
            predictions: Predicted probabilities (batch_size, num_classes)
            labels: Ground truth labels (batch_size, num_classes)
                    1: positive, 0: negative, -1: ignore
        
        Returns:
            Loss value
        """
        # Create mask (ignore labels == -1)
        mask = (labels != -1).float()
        
        # Convert labels to 0/1
        binary_labels = torch.clamp(labels, 0, 1)
        
        # Compute BCE loss
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
        
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            predictions = self.model(input_ids, attention_mask, self.edge_index)
            
            # Compute loss
            loss = self.compute_loss(predictions, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
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
                
                # Forward pass
                predictions = self.model(input_ids, attention_mask, self.edge_index)
                
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
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }, save_path)
    
    def load_model(self, filename: str):
        """Load model checkpoint"""
        load_path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(load_path)
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

