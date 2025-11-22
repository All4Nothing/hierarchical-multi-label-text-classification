"""
Stage 1: Document-Class Similarity Calculation using Textual Entailment
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple
from tqdm import tqdm
import os
import pickle


class DocumentClassSimilarity:
    """Calculate document-class similarity using textual entailment"""
    
    def __init__(
        self,
        model_name: str = "roberta-large-mnli",
        hypothesis_template: str = "This document is about {class_name}",
        device: str = "cuda",
        batch_size: int = 16,
        max_length: int = 256,
        cache_dir: str = "./cache"
    ):
        """
        Initialize similarity calculator
        
        Args:
            model_name: Pretrained NLI model name
            hypothesis_template: Template for hypothesis generation
            device: Device to run model on
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            cache_dir: Directory to cache results
        """
        self.model_name = model_name
        self.hypothesis_template = hypothesis_template
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        print(f"Model loaded on {device}")
    
    def _create_hypothesis(self, class_name: str) -> str:
        """Create hypothesis from class name"""
        # Replace underscores with spaces for better readability
        formatted_name = class_name.replace('_', ' ')
        return self.hypothesis_template.format(class_name=formatted_name)
    
    def calculate_entailment_score(
        self,
        premises: List[str],
        hypothesis: str
    ) -> np.ndarray:
        """
        Calculate entailment scores for a batch of premises with one hypothesis
        
        Args:
            premises: List of premise texts (documents)
            hypothesis: Hypothesis text
        
        Returns:
            Array of entailment probabilities
        """
        scores = []
        
        # Process in batches
        for i in range(0, len(premises), self.batch_size):
            batch_premises = premises[i:i + self.batch_size]
            
            # Create pairs
            pairs = [[premise, hypothesis] for premise in batch_premises]
            
            # Tokenize
            encodings = self.tokenizer(
                pairs,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            encodings = {k: v.to(self.device) for k, v in encodings.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Extract entailment probability (index 2 for RoBERTa-MNLI)
                # Labels: 0=contradiction, 1=neutral, 2=entailment
                entailment_probs = probs[:, 2].cpu().numpy()
                scores.extend(entailment_probs)
        
        return np.array(scores)
    
    def compute_similarity_matrix(
        self,
        documents: List[str],
        class_names: Dict[int, str],
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Compute similarity matrix for all document-class pairs
        
        Args:
            documents: List of document texts
            class_names: Dictionary mapping class_id to class_name
            use_cache: Whether to use cached results
        
        Returns:
            Similarity matrix (num_docs, num_classes)
        """
        num_docs = len(documents)
        num_classes = len(class_names)
        
        # Check cache
        cache_file = os.path.join(
            self.cache_dir,
            f"similarity_matrix_{num_docs}docs_{num_classes}classes.pkl"
        )
        
        if use_cache and os.path.exists(cache_file):
            print(f"Loading cached similarity matrix from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"Computing similarity matrix for {num_docs} documents and {num_classes} classes")
        
        # Initialize similarity matrix
        similarity_matrix = np.zeros((num_docs, num_classes), dtype=np.float32)
        
        # Compute similarities for each class
        for class_id, class_name in tqdm(class_names.items(), desc="Computing similarities"):
            # Create hypothesis
            hypothesis = self._create_hypothesis(class_name)
            
            # Calculate entailment scores for all documents
            scores = self.calculate_entailment_score(documents, hypothesis)
            
            # Store in matrix
            similarity_matrix[:, class_id] = scores
        
        # Save to cache
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(similarity_matrix, f)
        print(f"Saved similarity matrix to {cache_file}")
        
        return similarity_matrix
    
    def get_top_k_classes(
        self,
        doc_id: int,
        similarity_matrix: np.ndarray,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Get top-k most similar classes for a document
        
        Args:
            doc_id: Document ID
            similarity_matrix: Similarity matrix
            k: Number of top classes to return
        
        Returns:
            List of (class_id, similarity_score) tuples
        """
        scores = similarity_matrix[doc_id]
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        return [(idx, scores[idx]) for idx in top_k_indices]


class FastSimilarityCalculator:
    """
    Faster similarity calculation using class name embeddings
    Alternative to full entailment model for efficiency
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda",
        batch_size: int = 32,
        cache_dir: str = "./cache"
    ):
        """
        Initialize fast similarity calculator using sentence embeddings
        
        Args:
            model_name: Sentence transformer model
            device: Device to run on
            batch_size: Batch size
            cache_dir: Directory to cache results
        """
        from sentence_transformers import SentenceTransformer
        
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        
        print(f"Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
    
    def compute_similarity_matrix(
        self,
        documents: List[str],
        class_names: Dict[int, str],
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Compute cosine similarity between documents and class names
        
        Args:
            documents: List of document texts
            class_names: Dictionary of class names
            use_cache: Whether to use cached results
        
        Returns:
            Similarity matrix (num_docs, num_classes)
        """
        num_docs = len(documents)
        num_classes = len(class_names)
        
        # Check cache
        cache_file = os.path.join(
            self.cache_dir,
            f"fast_similarity_matrix_{num_docs}docs_{num_classes}classes.pkl"
        )
        
        if use_cache and os.path.exists(cache_file):
            print(f"Loading cached similarity matrix from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"Encoding {num_docs} documents...")
        doc_embeddings = self.model.encode(
            documents,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device
        )
        
        print(f"Encoding {num_classes} class names...")
        class_texts = [class_names[i].replace('_', ' ') for i in range(num_classes)]
        class_embeddings = self.model.encode(
            class_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device
        )
        
        # Compute cosine similarity
        print("Computing similarity matrix...")
        similarity_matrix = torch.nn.functional.cosine_similarity(
            doc_embeddings.unsqueeze(1),
            class_embeddings.unsqueeze(0),
            dim=2
        )
        
        similarity_matrix = similarity_matrix.cpu().numpy()
        
        # Save to cache
        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(similarity_matrix, f)
            print(f"Saved similarity matrix to {cache_file}")
        
        return similarity_matrix

