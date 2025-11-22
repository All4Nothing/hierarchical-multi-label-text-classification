"""
Data Loading and Preprocessing
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from torch.utils.data import Dataset
import torch


class DocumentCorpus:
    """Load and manage document corpus"""
    
    def __init__(self, corpus_file: str):
        """
        Load corpus file
        
        Args:
            corpus_file: Path to corpus file (train_corpus.txt or test_corpus.txt)
        """
        self.corpus_file = corpus_file
        self.documents = []
        self.doc_ids = []
        self.labels = []  # Ground truth labels if available
        
        self._load_corpus()
    
    def _load_corpus(self):
        """Load corpus from file"""
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    label = int(parts[0])
                    text = parts[1]
                    
                    self.doc_ids.append(idx)
                    self.documents.append(text)
                    self.labels.append(label)
        
        print(f"Loaded {len(self.documents)} documents from {self.corpus_file}")
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        return {
            'doc_id': self.doc_ids[idx],
            'text': self.documents[idx],
            'label': self.labels[idx]
        }
    
    def get_text(self, doc_id: int) -> str:
        """Get document text by ID"""
        return self.documents[doc_id]
    
    def get_all_texts(self) -> List[str]:
        """Get all document texts"""
        return self.documents
    
    def get_all_labels(self) -> List[int]:
        """Get all labels"""
        return self.labels


class TaxoDataset(Dataset):
    """PyTorch Dataset for TaxoClass"""
    
    def __init__(
        self,
        documents: List[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int = 256
    ):
        """
        Args:
            documents: List of document texts
            labels: Multi-label binary matrix (num_docs, num_classes)
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.documents = documents
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        text = self.documents[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.FloatTensor(label)
        }


def load_class_keywords(keywords_file: str) -> Dict[str, List[str]]:
    """
    Load class-related keywords
    
    Args:
        keywords_file: Path to class_related_keywords.txt
    
    Returns:
        Dictionary mapping class name to list of keywords
    """
    keywords_dict = {}
    
    with open(keywords_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(':', 1)
            if len(parts) == 2:
                class_name = parts[0]
                keywords = [kw.strip() for kw in parts[1].split(',')]
                keywords_dict[class_name] = keywords
    
    return keywords_dict


def create_multi_label_matrix(
    doc_labels: List[int],
    core_class_assignments: Dict[int, int],
    hierarchy,
    num_classes: int
) -> np.ndarray:
    """
    Create multi-label binary matrix
    
    Args:
        doc_labels: Ground truth labels for evaluation
        core_class_assignments: Dict mapping doc_id to core_class_id
        hierarchy: TaxonomyHierarchy object
        num_classes: Total number of classes
    
    Returns:
        Binary matrix (num_docs, num_classes)
        1: positive, 0: negative, -1: ignore (descendants)
    """
    num_docs = len(doc_labels)
    
    # Validate and adjust num_classes if needed
    if core_class_assignments:
        max_core_class = max(core_class_assignments.values())
        if max_core_class >= num_classes:
            actual_num_classes = max_core_class + 1
            print(f"⚠️  Warning: Max core class ID ({max_core_class}) >= num_classes ({num_classes})")
            print(f"   Adjusting num_classes to {actual_num_classes}")
            num_classes = actual_num_classes
    
    label_matrix = np.zeros((num_docs, num_classes), dtype=np.float32)
    
    for doc_id, core_class in core_class_assignments.items():
        if doc_id >= num_docs:
            continue
        
        # Validate core_class is within bounds
        if core_class < 0 or core_class >= num_classes:
            print(f"⚠️  Warning: Document {doc_id} has invalid core class ID {core_class} (valid range: 0-{num_classes-1}), skipping...")
            continue
        
        # Positive: core class + ancestors
        positive_classes = {core_class}
        try:
            ancestors = hierarchy.get_ancestors(core_class)
            positive_classes.update(ancestors)
        except (KeyError, AttributeError):
            # If core_class not in hierarchy, skip ancestors
            pass
        
        # Descendants (ignore)
        try:
            descendants = hierarchy.get_descendants(core_class)
        except (KeyError, AttributeError):
            descendants = set()
        
        # Set labels
        for class_id in range(num_classes):
            if class_id in positive_classes:
                label_matrix[doc_id, class_id] = 1.0
            elif class_id in descendants:
                label_matrix[doc_id, class_id] = -1.0  # Ignore
            else:
                label_matrix[doc_id, class_id] = 0.0  # Negative
    
    return label_matrix


def create_ground_truth_matrix(
    doc_labels: List[int],
    hierarchy,
    num_classes: int
) -> np.ndarray:
    """
    Create ground truth multi-label matrix for evaluation
    
    Args:
        doc_labels: List of ground truth class labels
        hierarchy: TaxonomyHierarchy object
        num_classes: Total number of classes
    
    Returns:
        Binary matrix (num_docs, num_classes)
    """
    num_docs = len(doc_labels)
    
    # Check if class IDs are within valid range
    if doc_labels:
        max_class_id = max(doc_labels)
        min_class_id = min(doc_labels)
        
        # Adjust num_classes if needed (handle 0-based vs 1-based indexing)
        if max_class_id >= num_classes:
            # If max_class_id is 531 and num_classes is 531, we need 532 (0-531)
            # Or if class IDs are 1-based (1-531), we need to adjust
            actual_num_classes = max_class_id + 1
            print(f"⚠️  Warning: Max class ID ({max_class_id}) >= num_classes ({num_classes})")
            print(f"   Adjusting num_classes to {actual_num_classes}")
            num_classes = actual_num_classes
        elif min_class_id < 0:
            print(f"⚠️  Warning: Min class ID ({min_class_id}) < 0, adjusting...")
            # Shift all IDs to be non-negative
            shift = -min_class_id
            doc_labels = [label + shift for label in doc_labels]
            num_classes += shift
            print(f"   Shifted class IDs by {shift}, new num_classes: {num_classes}")
    
    gt_matrix = np.zeros((num_docs, num_classes), dtype=np.int32)
    
    for doc_id, label_class in enumerate(doc_labels):
        # Validate class ID is within bounds
        if label_class < 0 or label_class >= num_classes:
            print(f"⚠️  Warning: Document {doc_id} has invalid class ID {label_class} (valid range: 0-{num_classes-1}), skipping...")
            continue
        
        # Mark the label and all its ancestors as positive
        gt_matrix[doc_id, label_class] = 1
        try:
            for ancestor in hierarchy.get_ancestors(label_class):
                if 0 <= ancestor < num_classes:
                    gt_matrix[doc_id, ancestor] = 1
        except (KeyError, AttributeError):
            # If class ID not in hierarchy, skip ancestors
            pass
    
    return gt_matrix

