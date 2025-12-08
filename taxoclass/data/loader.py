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
    
    def __init__(self, corpus_file: str, has_labels: bool = False):
        """
        Load corpus file
        
        Args:
            corpus_file: Path to corpus file (train_corpus.txt or test_corpus.txt)
        """
        self.corpus_file = corpus_file
        self.documents = []
        self.doc_ids = []
        self.labels = []  # Ground truth labels if available
        self.has_labels = has_labels
        
        self._load_corpus()
    
    def _load_corpus(self):
        """Load corpus from file"""
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    # has_labels 여부에 따라 다르게 처리
                    if self.has_labels:
                        # 정답이 있는 파일인 경우 (예: 검증셋 등)
                        id_or_label = int(parts[0])
                        text = parts[1]
                        self.doc_ids.append(idx)
                        self.labels.append(id_or_label)
                    else:
                        # 정답이 없는 학습용 코퍼스인 경우 (과제 데이터)
                        doc_id = int(parts[0]) # 첫 컬럼은 문서 번호
                        text = parts[1]
                        self.doc_ids.append(doc_id)
                        self.labels.append(-1) # 정답 없음(-1)으로 표시
                    
                    # Note: doc_ids는 위에서 이미 추가했으므로 여기서는 documents만 추가
                    self.documents.append(text)
        
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
    core_class_assignments: Dict[int, List[int]],
    hierarchy,
    num_classes: int
) -> np.ndarray:
    """
    Create multi-label binary matrix from core class assignments
    
    Note: This function now delegates to create_training_labels from models.core_mining
    which properly implements the hierarchical label generation as per TaxoClass paper.
    
    Args:
        doc_labels: Ground truth labels for evaluation (not used, kept for compatibility)
        core_class_assignments: Dict mapping doc_id to list of core_class_ids (multi-label)
        hierarchy: TaxonomyHierarchy object
        num_classes: Total number of classes
    
    Returns:
        Binary matrix (num_docs, num_classes)
        1: positive (core class or ancestor)
        0: negative (other classes)
        -1: ignore (descendants of core classes)
    """
    from models.core_mining import create_training_labels
    
    num_docs = len(doc_labels)
    
    # Use the proper implementation from models.core_mining
    # Pass num_docs to ensure correct size
    label_matrix = create_training_labels(
        core_classes_dict=core_class_assignments,
        hierarchy=hierarchy,
        num_classes=num_classes,
        num_docs=num_docs
    )
    
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
    # num_classes는 hierarchy에서 가져온 고정값이어야 함
# 범위를 벗어나는 라벨이 있어도 num_classes를 늘리지 않고, 해당 라벨을 무시함

    if doc_labels:
        max_label = max(doc_labels)
        min_label = min(doc_labels)
        
        # Check if all labels are -1 (test corpus without ground truth)
        all_negative = all(label == -1 for label in doc_labels)
        
        if all_negative:
            # Test corpus without ground truth - this is expected, no warning needed
            # Return zero matrix (no ground truth available)
            return np.zeros((num_docs, num_classes), dtype=np.int32)
        
        # 디버깅용 경고 출력 (필요시 주석 처리)
        if max_label >= num_classes:
            print(f"⚠️  Warning: Found label ID {max_label} >= num_classes ({num_classes}). This label will be IGNORED.")
        if min_label < 0 and not all_negative:
            # Only warn if there are mixed negative and non-negative labels (unexpected case)
            print(f"⚠️  Warning: Found negative label ID {min_label}. This label will be IGNORED.")
    
    gt_matrix = np.zeros((num_docs, num_classes), dtype=np.int32)
    
    # 실제 행렬 생성 루프
    for doc_id, label_class in enumerate(doc_labels):
        # [핵심] 유효 범위(0 ~ num_classes-1) 내의 라벨만 1로 표시
        if 0 <= label_class < num_classes:
            # label_class와 그 조상들을 1로 설정
            gt_matrix[doc_id, label_class] = 1
            try:
                for ancestor in hierarchy.get_ancestors(label_class):
                    if 0 <= ancestor < num_classes: # 조상 ID도 유효한지 체크
                        gt_matrix[doc_id, ancestor] = 1
            except (KeyError, AttributeError):
                pass
        else:
            # 범위를 벗어난 라벨은 조용히 무시 (Skip)
            continue
    
    return gt_matrix

