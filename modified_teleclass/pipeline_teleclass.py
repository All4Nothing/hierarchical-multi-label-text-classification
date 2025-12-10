"""
Hierarchical Multi-Label Text Classification Pipeline
Based on TELEClass framework with transductive learning.

This pipeline uses BOTH train and test corpora during the unsupervised 
representation learning and refinement phases to maximize cluster density.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = 'NO'

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification, get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Set
from tqdm import tqdm
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PHASE 0: REPRODUCIBILITY
# ============================================================================

def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries.
    This is CRITICAL for competition submissions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")


# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    """
    Loads and parses all competition data files.
    Handles corpora, taxonomy, keywords, and class definitions.
    """
    
    def __init__(self, data_dir: str = "Amazon_products"):
        self.data_dir = data_dir
        self.train_corpus = []
        self.test_corpus = []
        self.all_corpus = []
        self.train_indices = []
        self.test_indices = []
        self.hierarchy_graph = nx.DiGraph()
        self.class_keywords = {}
        self.all_classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
    def load_all(self):
        """Load all data files."""
        logger.info("Loading all data files...")
        self._load_corpora()
        self._load_taxonomy()
        self._load_keywords()
        self._load_classes()
        logger.info(f"Data loading complete:")
        logger.info(f"  - Train corpus: {len(self.train_corpus)} documents")
        logger.info(f"  - Test corpus: {len(self.test_corpus)} documents")
        logger.info(f"  - Combined corpus: {len(self.all_corpus)} documents")
        logger.info(f"  - Classes: {len(self.all_classes)}")
        logger.info(f"  - Hierarchy edges: {self.hierarchy_graph.number_of_edges()}")
        
    def _load_corpora(self):
        """
        Load train and test corpora, combine them into all_corpus.
        Keep track of indices for later separation.
        """
        # Load train corpus
        train_path = os.path.join(self.data_dir, "train", "train_corpus.txt")
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    doc_id, text = parts
                    self.train_corpus.append(text)
                    
        # Load test corpus
        test_path = os.path.join(self.data_dir, "test", "test_corpus.txt")
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    doc_id, text = parts
                    self.test_corpus.append(text)
        
        # Combine into single corpus for transductive learning
        # CRITICAL: This allows the model to leverage test data distribution
        # during the unsupervised phases without seeing labels
        self.all_corpus = self.train_corpus + self.test_corpus
        self.train_indices = list(range(len(self.train_corpus)))
        self.test_indices = list(range(len(self.train_corpus), len(self.all_corpus)))
        
    def _load_taxonomy(self):
        """
        Load class hierarchy and build NetworkX DiGraph.
        Format: parent_id \t child_id
        """
        hierarchy_path = os.path.join(self.data_dir, "class_hierarchy.txt")
        with open(hierarchy_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    parent_id, child_id = parts
                    self.hierarchy_graph.add_edge(int(parent_id), int(child_id))
                    
    def _load_keywords(self):
        """
        Load class keywords.
        Format: class_name : keyword1,keyword2,...
        """
        keywords_path = os.path.join(self.data_dir, "class_related_keywords.txt")
        with open(keywords_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(':', 1)
                if len(parts) == 2:
                    class_name, keywords = parts
                    keyword_list = [kw.strip() for kw in keywords.split(',')]
                    self.class_keywords[class_name] = keyword_list
                    
    def _load_classes(self):
        """
        Load class list and create mappings.
        Format: class_id \t class_name
        """
        classes_path = os.path.join(self.data_dir, "classes.txt")
        with open(classes_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    class_id, class_name = parts
                    self.all_classes.append(class_name)
                    self.class_to_idx[class_name] = int(class_id)
                    self.idx_to_class[int(class_id)] = class_name


# ============================================================================
# PHASE 1: INITIALIZATION - CLASS REPRESENTATION VIA CONTEXTUAL INJECTION
# ============================================================================

class ClassRepresentationModule:
    """
    Creates contextualized class embeddings by injecting keywords.
    Uses sentence-transformers for high-quality semantic representations.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}")
        
    def create_class_descriptions(self, class_keywords: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Create natural language descriptions for each class.
        Format: "The product category is {class_name}, associated with keywords: {keywords}"
        """
        descriptions = {}
        for class_name, keywords in class_keywords.items():
            keyword_str = ", ".join(keywords)
            description = f"The product category is {class_name}, associated with keywords: {keyword_str}"
            descriptions[class_name] = description
        return descriptions
    
    def encode_classes(self, class_descriptions: Dict[str, str], all_classes: List[str]) -> torch.Tensor:
        """
        Encode class descriptions into embeddings.
        Returns: [num_classes, embedding_dim]
        """
        logger.info("Encoding class descriptions...")
        # Ensure ordering matches all_classes
        descriptions_list = [class_descriptions.get(cls, f"The product category is {cls}") 
                            for cls in all_classes]
        embeddings = self.model.encode(
            descriptions_list,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=self.device
        )
        return embeddings
    
    def encode_documents(self, documents: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode all documents (train + test combined).
        Returns: [num_docs, embedding_dim]
        """
        logger.info(f"Encoding {len(documents)} documents...")
        embeddings = self.model.encode(
            documents,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=batch_size,
            device=self.device
        )
        return embeddings


# ============================================================================
# PHASE 2: REFINEMENT LOOP - ITERATIVE PSEUDO-LABELING
# ============================================================================

class IterativePseudoLabeler:
    """
    Performs iterative refinement using transductive signal.
    Key insight: By including test data in the refinement loop,
    we align class centroids with the actual data distribution.
    """
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
    def compute_similarity(self, doc_embeddings: torch.Tensor, class_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between documents and classes.
        Returns: [num_docs, num_classes]
        """
        # Normalize embeddings
        doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)
        class_embeddings = torch.nn.functional.normalize(class_embeddings, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.mm(doc_embeddings, class_embeddings.t())
        return similarity
    
    def refine_class_embeddings(
        self,
        doc_embeddings: torch.Tensor,
        class_embeddings: torch.Tensor,
        num_iterations: int = 3,
        top_n_reliable: int = 20,
        initial_top_k: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Iterative refinement loop:
        1. Calculate initial similarities
        2. For each iteration:
            a. Select top-N most confident documents for each class
            b. Update class embeddings as centroids of reliable documents
        3. Return final class embeddings and similarity scores
        
        CRITICAL: This operates on ALL documents (train + test combined),
        which is the key to transductive learning.
        """
        logger.info(f"Starting iterative refinement for {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            
            # Compute similarity scores
            similarity = self.compute_similarity(doc_embeddings, class_embeddings)
            
            # For each class, find top-N most confident documents
            new_class_embeddings = []
            for class_idx in range(class_embeddings.shape[0]):
                class_scores = similarity[:, class_idx]
                
                # Get top-N reliable documents
                top_n_indices = torch.topk(class_scores, min(top_n_reliable, len(class_scores))).indices
                
                # Calculate new class embedding as centroid
                reliable_embeddings = doc_embeddings[top_n_indices]
                new_class_embedding = reliable_embeddings.mean(dim=0)
                new_class_embeddings.append(new_class_embedding)
            
            # Update class embeddings
            class_embeddings = torch.stack(new_class_embeddings)
            
            logger.info(f"  Class embeddings updated based on reliable document centroids")
        
        # Final similarity computation
        final_similarity = self.compute_similarity(doc_embeddings, class_embeddings)
        
        return class_embeddings, final_similarity
    
    def assign_labels_with_gap(
        self,
        similarity: torch.Tensor,
        min_labels: int = 2,
        max_gap_search: int = 5
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Assign pseudo-labels using similarity gap heuristic.
        
        Logic:
        1. Sort classes by similarity for each document
        2. Find the largest gap in top-5 similarities
        3. Assign labels up to the gap (minimum 2 labels)
        
        Returns: (pseudo_labels, pseudo_scores)
        """
        logger.info("Assigning pseudo-labels with gap-based cutoff...")
        
        pseudo_labels = []
        pseudo_scores = []
        
        for doc_idx in range(similarity.shape[0]):
            scores = similarity[doc_idx]
            
            # Sort in descending order
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            
            # Calculate gaps between consecutive scores
            diffs = sorted_scores[:-1] - sorted_scores[1:]
            
            # Look for largest gap in top positions
            valid_range = diffs[:max_gap_search]
            best_gap_idx = torch.argmax(valid_range).item()
            
            # Determine number of labels (minimum 2)
            num_labels = max(min_labels, best_gap_idx + 1)
            
            # Assign labels
            selected_indices = sorted_indices[:num_labels].cpu().tolist()
            selected_scores = sorted_scores[:num_labels].cpu().tolist()
            
            pseudo_labels.append(selected_indices)
            pseudo_scores.append(selected_scores)
        
        # Log statistics
        avg_labels = np.mean([len(labels) for labels in pseudo_labels])
        logger.info(f"  Average labels per document: {avg_labels:.2f}")
        
        return pseudo_labels, pseudo_scores


# ============================================================================
# PHASE 3: AUGMENTATION - LLM GENERATION FOR STARVED CLASSES
# ============================================================================

class AugmentationModule:
    """
    Handles data augmentation for under-represented classes.
    In production, this would use LLM to generate synthetic reviews.
    """
    
    def __init__(self, data_loader: DataLoader, augmentation_dir: str = "augmented_data"):
        self.data_loader = data_loader
        self.augmentation_dir = augmentation_dir
        os.makedirs(augmentation_dir, exist_ok=True)
        
    def identify_starved_classes(
        self,
        pseudo_labels: List[List[int]],
        train_indices: List[int],
        threshold: int = 10
    ) -> List[int]:
        """
        Identify classes with fewer than threshold assigned documents.
        Only count training documents (not test) for fair evaluation.
        """
        class_counts = {}
        for idx in train_indices:
            for class_idx in pseudo_labels[idx]:
                class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
        
        starved_classes = [
            class_idx for class_idx in range(len(self.data_loader.all_classes))
            if class_counts.get(class_idx, 0) < threshold
        ]
        
        logger.info(f"Identified {len(starved_classes)} starved classes (< {threshold} documents)")
        return starved_classes
    
    def generate_augmentation_data(
        self,
        starved_classes: List[int],
        num_samples_per_class: int = 20
    ) -> Tuple[List[str], List[List[int]]]:
        """
        Placeholder for LLM-based augmentation.
        
        In production, this would:
        1. For each starved class, get its taxonomy path
        2. Use LLM to generate synthetic product reviews
        3. Save to JSON and load back
        
        For now, we return empty lists (augmentation is optional).
        """
        logger.info("Augmentation module called (placeholder - no synthetic data generated)")
        
        # In production implementation:
        # augmented_texts = []
        # augmented_labels = []
        # for class_idx in starved_classes:
        #     class_name = self.data_loader.idx_to_class[class_idx]
        #     taxonomy_path = self._get_taxonomy_path(class_idx)
        #     synthetic_reviews = self._call_llm_api(class_name, taxonomy_path, num_samples_per_class)
        #     augmented_texts.extend(synthetic_reviews)
        #     augmented_labels.extend([[class_idx]] * len(synthetic_reviews))
        
        return [], []
    
    def _get_taxonomy_path(self, class_idx: int) -> str:
        """Get full taxonomy path for a class (e.g., 'Baby Product > Feeding > Baby Cereal')"""
        # Use NetworkX to find path from root
        # This is a simplified version
        class_name = self.data_loader.idx_to_class[class_idx]
        return class_name


# ============================================================================
# PHASE 4: POST-PROCESSING - HIERARCHY EXPANSION
# ============================================================================

class HierarchyExpander:
    """
    Expands pseudo-labels to include all ancestor classes.
    If a document is labeled 'Baby Cereal', also add 'Feeding' and 'Baby Product'.
    """
    
    def __init__(self, hierarchy_graph: nx.DiGraph, class_to_idx: Dict[str, int]):
        self.hierarchy_graph = hierarchy_graph
        self.class_to_idx = class_to_idx
        
    def get_ancestors(self, class_idx: int) -> Set[int]:
        """Get all ancestor classes for a given class."""
        ancestors = set()
        
        # BFS to find all ancestors
        queue = [class_idx]
        visited = {class_idx}
        
        while queue:
            current = queue.pop(0)
            
            # Find parents in the hierarchy graph
            for node in self.hierarchy_graph.nodes():
                if self.hierarchy_graph.has_edge(node, current) and node not in visited:
                    ancestors.add(node)
                    queue.append(node)
                    visited.add(node)
        
        return ancestors
    
    def expand_labels_with_hierarchy(
        self,
        pseudo_labels: List[List[int]]
    ) -> List[List[int]]:
        """
        Expand all pseudo-labels to include ancestors.
        This enforces hierarchical consistency.
        """
        logger.info("Expanding labels with hierarchy...")
        
        expanded_labels = []
        for doc_labels in tqdm(pseudo_labels, desc="Expanding labels"):
            expanded = set(doc_labels)
            
            # Add all ancestors for each label
            for class_idx in doc_labels:
                ancestors = self.get_ancestors(class_idx)
                expanded.update(ancestors)
            
            expanded_labels.append(sorted(list(expanded)))
        
        # Log statistics
        avg_original = np.mean([len(labels) for labels in pseudo_labels])
        avg_expanded = np.mean([len(labels) for labels in expanded_labels])
        logger.info(f"  Average labels: {avg_original:.2f} -> {avg_expanded:.2f}")
        
        return expanded_labels


# ============================================================================
# PHASE 5: CLASSIFIER TRAINING
# ============================================================================

class MultiLabelDataset(Dataset):
    """PyTorch Dataset for multi-label classification."""
    
    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label_indices = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create multi-label target
        target = torch.zeros(531)  # num_classes
        target[label_indices] = 1.0
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': target
        }


class BERTClassifierTrainer:
    """
    Trains a BERT-based multi-label classifier.
    Uses pseudo-labels from the refinement phase.
    """
    
    def __init__(
        self,
        num_classes: int = 531,
        model_name: str = "bert-base-uncased",
        device: str = None
    ):
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing BERT classifier: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        
    def prepare_data(
        self,
        train_texts: List[str],
        train_labels: List[List[int]],
        val_texts: List[str] = None,
        val_labels: List[List[int]] = None,
        batch_size: int = 16,
        max_length: int = 128
    ):
        """Prepare DataLoaders."""
        train_dataset = MultiLabelDataset(train_texts, train_labels, self.tokenizer, max_length)
        self.train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_texts is not None and val_labels is not None:
            val_dataset = MultiLabelDataset(val_texts, val_labels, self.tokenizer, max_length)
            self.val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            self.val_loader = None
            
        logger.info(f"Prepared {len(train_dataset)} training samples")
        if self.val_loader:
            logger.info(f"Prepared {len(val_dataset)} validation samples")
    
    def train(
        self,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        output_dir: str = "outputs/models"
    ):
        """Train the classifier."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            problem_type="multi_label_classification"
        )
        self.model.to(self.device)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            self.model.train()
            train_loss = 0
            
            for batch in tqdm(self.train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                loss = criterion(logits, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(self.train_loader)
            logger.info(f"  Training loss: {avg_train_loss:.4f}")
            
            # Validation
            if self.val_loader:
                val_loss = self._validate(criterion)
                logger.info(f"  Validation loss: {val_loss:.4f}")
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_path = os.path.join(output_dir, "best_model")
                    self.model.save_pretrained(save_path)
                    self.tokenizer.save_pretrained(save_path)
                    logger.info(f"  Saved best model to {save_path}")
        
        # Save final model
        final_path = os.path.join(output_dir, "final_model")
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        logger.info(f"Training complete. Final model saved to {final_path}")
    
    def _validate(self, criterion):
        """Run validation."""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
        
        return val_loss / len(self.val_loader)


# ============================================================================
# PHASE 6: INFERENCE AND SUBMISSION GENERATION
# ============================================================================

class InferenceModule:
    """
    Performs inference on test set and generates Kaggle submission.
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = None,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model from {model_path}")
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        tokenizer_path = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
    def predict(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 128,
        threshold: float = 0.5
    ) -> List[List[int]]:
        """
        Predict labels for texts.
        Returns: List of label indices for each text.
        """
        logger.info(f"Running inference on {len(texts)} documents...")
        
        predictions = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encodings = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.sigmoid(logits)
            
            # Threshold to get labels
            batch_predictions = (probs > threshold).cpu().numpy()
            
            for pred in batch_predictions:
                label_indices = np.where(pred)[0].tolist()
                # Ensure at least one label
                if not label_indices:
                    label_indices = [np.argmax(probs[len(predictions)].cpu().numpy())]
                predictions.append(label_indices)
        
        return predictions
    
    def generate_submission(
        self,
        predictions: List[List[int]],
        idx_to_class: Dict[int, str],
        output_path: str = "submission.csv"
    ):
        """
        Generate Kaggle submission file.
        Format: ID, Label (space-separated class names)
        """
        logger.info(f"Generating submission file: {output_path}")
        
        submission_data = []
        for doc_id, label_indices in enumerate(predictions):
            label_str = " ".join([str(idx) for idx in label_indices])
            submission_data.append({'id': doc_id, 'labels': label_str})

        df = pd.DataFrame(submission_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Submission saved: {output_path}")
        logger.info(f"  Total predictions: {len(predictions)}")
        logger.info(f"  Average labels per document: {np.mean([len(p) for p in predictions]):.2f}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class TELEClassPipeline:
    """
    Main pipeline orchestrator.
    Executes all phases in sequence.
    """
    
    def __init__(self, data_dir: str = "Amazon_products", output_dir: str = "outputs", seed: int = 42):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.seed = seed
        
        # Set seed first
        set_seed(seed)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "intermediate"), exist_ok=True)
        
    def run(self):
        """Execute the full pipeline."""
        logger.info("="*80)
        logger.info("TELECLASS PIPELINE - HIERARCHICAL MULTI-LABEL CLASSIFICATION")
        logger.info("="*80)
        
        # Load data
        logger.info("\n" + "="*80)
        logger.info("LOADING DATA")
        logger.info("="*80)
        data_loader = DataLoader(self.data_dir)
        data_loader.load_all()
        
        # Phase 1: Class Representation
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: CLASS REPRESENTATION VIA CONTEXTUAL INJECTION")
        logger.info("="*80)
        class_repr = ClassRepresentationModule()
        class_descriptions = class_repr.create_class_descriptions(data_loader.class_keywords)
        class_embeddings = class_repr.encode_classes(class_descriptions, data_loader.all_classes)
        
        # Encode ALL documents (train + test combined for transductive learning)
        doc_embeddings = class_repr.encode_documents(data_loader.all_corpus)
        
        logger.info(f"Class embeddings shape: {class_embeddings.shape}")
        logger.info(f"Document embeddings shape: {doc_embeddings.shape}")
        
        # Phase 2: Iterative Refinement
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: ITERATIVE PSEUDO-LABELING (TRANSDUCTIVE)")
        logger.info("="*80)
        logger.info("Using BOTH train and test corpora for refinement...")
        
        labeler = IterativePseudoLabeler()
        refined_class_embeddings, final_similarity = labeler.refine_class_embeddings(
            doc_embeddings,
            class_embeddings,
            num_iterations=3,
            top_n_reliable=20
        )
        
        pseudo_labels, pseudo_scores = labeler.assign_labels_with_gap(
            final_similarity,
            min_labels=2
        )
        
        # Save intermediate results
        torch.save({
            'pseudo_labels': pseudo_labels,
            'pseudo_scores': pseudo_scores,
            'class_embeddings': refined_class_embeddings,
            'doc_embeddings': doc_embeddings
        }, os.path.join(self.output_dir, "intermediate", "phase2_outputs.pt"))
        
        # Phase 3: Augmentation (optional)
        logger.info("\n" + "="*80)
        logger.info("PHASE 3: DATA AUGMENTATION FOR STARVED CLASSES")
        logger.info("="*80)
        
        aug_module = AugmentationModule(data_loader)
        starved_classes = aug_module.identify_starved_classes(
            pseudo_labels,
            data_loader.train_indices,
            threshold=10
        )
        augmented_texts, augmented_labels = aug_module.generate_augmentation_data(starved_classes)
        
        # Phase 4: Hierarchy Expansion
        logger.info("\n" + "="*80)
        logger.info("PHASE 4: HIERARCHY EXPANSION")
        logger.info("="*80)
        
        expander = HierarchyExpander(data_loader.hierarchy_graph, data_loader.class_to_idx)
        expanded_labels = expander.expand_labels_with_hierarchy(pseudo_labels)

        avg_labels = np.mean([len(l) for l in expanded_train_labels])
        print(f"Average Labels after Expansion: {avg_labels}")
        
        # Phase 5: Classifier Training
        logger.info("\n" + "="*80)
        logger.info("PHASE 5: BERT CLASSIFIER TRAINING")
        logger.info("="*80)
        
        # Prepare training data (only train corpus + augmentation)
        train_texts = data_loader.train_corpus + augmented_texts
        train_labels = [expanded_labels[i] for i in data_loader.train_indices] + augmented_labels
        
        logger.info(f"Training set size: {len(train_texts)} documents")
        
        trainer = BERTClassifierTrainer(num_classes=len(data_loader.all_classes))
        trainer.prepare_data(train_texts, train_labels, batch_size=16)
        trainer.train(
            num_epochs=10,
            learning_rate=2e-5,
            output_dir=os.path.join(self.output_dir, "models")
        )
        
        # Phase 6: Inference
        logger.info("\n" + "="*80)
        logger.info("PHASE 6: INFERENCE AND SUBMISSION GENERATION")
        logger.info("="*80)
        
        inference = InferenceModule(
            model_path=os.path.join(self.output_dir, "models", "best_model")
        )
        
        test_predictions = inference.predict(
            data_loader.test_corpus,
            batch_size=32,
            threshold=0.5
        )
        
        # Apply hierarchy expansion to predictions
        test_predictions_expanded = expander.expand_labels_with_hierarchy(test_predictions)
        
        # Generate submission
        inference.generate_submission(
            test_predictions_expanded,
            data_loader.idx_to_class,
            output_path=os.path.join(self.output_dir, "submission.csv")
        )
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE!")
        logger.info("="*80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check if we're running from taxoclass directory or parent directory
    import sys
    if os.path.exists("Amazon_products"):
        data_dir = "Amazon_products"
    elif os.path.exists("../Amazon_products"):
        data_dir = "../Amazon_products"
    else:
        print("Error: Cannot find Amazon_products directory")
        print("Please run from the project root or taxoclass directory")
        sys.exit(1)
    
    pipeline = TELEClassPipeline(
        data_dir=data_dir,
        output_dir="outputs",
        seed=42
    )
    pipeline.run()
