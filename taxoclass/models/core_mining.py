"""
Stage 2: Document Core Class Mining
"""
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from tqdm import tqdm


class CoreClassMiner:
    """Mine core classes for documents using top-down search and confidence scoring"""
    
    def __init__(
        self,
        hierarchy,
        similarity_matrix: np.ndarray,
        candidate_power: int = 2,
        confidence_percentile: int = 50
    ):
        """
        Initialize core class miner
        
        Args:
            hierarchy: TaxonomyHierarchy object
            similarity_matrix: Document-class similarity matrix (num_docs, num_classes)
            candidate_power: Power for candidate selection (level+1)^power
            confidence_percentile: Percentile for confidence threshold (default: 50 for median)
        """
        self.hierarchy = hierarchy
        self.similarity_matrix = similarity_matrix
        self.candidate_power = candidate_power
        self.confidence_percentile = confidence_percentile
        
        self.num_docs = similarity_matrix.shape[0]
        self.num_classes = similarity_matrix.shape[1]
        
        # Store candidates and core classes
        self.candidate_classes = {}  # doc_id -> set of candidate classes
        self.core_classes = {}  # doc_id -> core class id
        self.confidence_scores = {}  # doc_id -> confidence score
        
        print(f"Initialized CoreClassMiner for {self.num_docs} docs and {self.num_classes} classes")
    
    def select_candidates_top_down(self, doc_id: int) -> Set[int]:
        """
        Select candidate classes using top-down search
        
        Args:
            doc_id: Document ID
        
        Returns:
            Set of candidate class IDs
        """
        candidates = set()
        doc_similarities = self.similarity_matrix[doc_id]
        
        # Start from root nodes
        roots = self.hierarchy.get_roots()
        current_level_nodes = set(roots)
        candidates.update(roots)
        
        # Get maximum level in hierarchy
        max_level = max(self.hierarchy.levels.values())
        
        # Top-down traversal
        for level in range(max_level):
            # Number of candidates to select at this level
            num_candidates = (level + 1) ** self.candidate_power
            
            # Get all children of current level nodes
            next_level_children = set()
            for node in current_level_nodes:
                children = self.hierarchy.get_children(node)
                next_level_children.update(children)
            
            if not next_level_children:
                break
            
            # Sort children by similarity and select top candidates
            children_list = list(next_level_children)
            children_similarities = [(child, doc_similarities[child]) for child in children_list]
            children_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Select top candidates
            top_candidates = [child for child, sim in children_similarities[:num_candidates]]
            candidates.update(top_candidates)
            current_level_nodes = set(top_candidates)
        
        return candidates
    
    def compute_confidence_score(self, doc_id: int, class_id: int) -> float:
        """
        Compute confidence score for a document-class pair
        
        conf(D, c) = sim(D, c) - max(sim(D, parent/siblings))
        
        Args:
            doc_id: Document ID
            class_id: Class ID
        
        Returns:
            Confidence score
        """
        doc_similarities = self.similarity_matrix[doc_id]
        sim_c = doc_similarities[class_id]
        
        # Get parents and siblings
        parents = self.hierarchy.get_parents(class_id)
        siblings = self.hierarchy.get_siblings(class_id)
        family = set(parents) | siblings
        
        if not family:
            # If no family members, confidence is just the similarity
            return sim_c
        
        # Maximum similarity among family members
        max_family_sim = max([doc_similarities[f] for f in family])
        
        # Confidence score
        confidence = sim_c - max_family_sim
        
        return confidence
    
    def compute_confidence_thresholds(self) -> Dict[int, float]:
        """
        Compute confidence threshold for each class based on median
        
        Returns:
            Dictionary mapping class_id to threshold
        """
        # Collect confidence scores for each class across all documents
        class_confidences = defaultdict(list)
        
        print("Computing confidence scores for all document-class pairs...")
        for doc_id in tqdm(range(self.num_docs)):
            candidates = self.candidate_classes.get(doc_id, set())
            
            for class_id in candidates:
                conf_score = self.compute_confidence_score(doc_id, class_id)
                class_confidences[class_id].append(conf_score)
        
        # Compute threshold (percentile) for each class
        thresholds = {}
        for class_id, scores in class_confidences.items():
            if scores:
                threshold = np.percentile(scores, self.confidence_percentile)
                thresholds[class_id] = threshold
            else:
                thresholds[class_id] = 0.0
        
        return thresholds
    
    def identify_core_classes(self) -> Dict[int, int]:
        """
        Identify core class for each document
        
        Returns:
            Dictionary mapping doc_id to core_class_id
        """
        print("Step 1: Selecting candidate classes using top-down search...")
        for doc_id in tqdm(range(self.num_docs)):
            candidates = self.select_candidates_top_down(doc_id)
            self.candidate_classes[doc_id] = candidates
        
        print("Step 2: Computing confidence thresholds...")
        thresholds = self.compute_confidence_thresholds()
        
        print("Step 3: Identifying core classes...")
        for doc_id in tqdm(range(self.num_docs)):
            candidates = self.candidate_classes[doc_id]
            
            # Compute confidence scores for all candidates
            candidate_scores = []
            for class_id in candidates:
                conf_score = self.compute_confidence_score(doc_id, class_id)
                threshold = thresholds.get(class_id, 0.0)
                
                # Check if confidence exceeds threshold
                if conf_score >= threshold:
                    candidate_scores.append((class_id, conf_score))
            
            # Select class with highest confidence as core class
            if candidate_scores:
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                core_class, core_conf = candidate_scores[0]
                
                self.core_classes[doc_id] = core_class
                self.confidence_scores[doc_id] = core_conf
            else:
                # Fallback: select class with highest similarity among candidates
                if candidates:
                    doc_similarities = self.similarity_matrix[doc_id]
                    best_class = max(candidates, key=lambda c: doc_similarities[c])
                    self.core_classes[doc_id] = best_class
                    self.confidence_scores[doc_id] = doc_similarities[best_class]
        
        print(f"Identified core classes for {len(self.core_classes)} documents")
        
        return self.core_classes
    
    def get_core_class(self, doc_id: int) -> int:
        """Get core class for a document"""
        return self.core_classes.get(doc_id, -1)
    
    def get_confidence_score(self, doc_id: int) -> float:
        """Get confidence score for a document"""
        return self.confidence_scores.get(doc_id, 0.0)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistics about core class mining"""
        stats = {
            'num_docs_with_core_class': len(self.core_classes),
            'avg_candidates_per_doc': np.mean([len(candidates) for candidates in self.candidate_classes.values()]),
            'avg_confidence_score': np.mean(list(self.confidence_scores.values())),
            'min_confidence_score': np.min(list(self.confidence_scores.values())) if self.confidence_scores else 0.0,
            'max_confidence_score': np.max(list(self.confidence_scores.values())) if self.confidence_scores else 0.0
        }
        
        return stats
    
    def analyze_core_class_distribution(self) -> Dict[int, int]:
        """
        Analyze distribution of core classes
        
        Returns:
            Dictionary mapping class_id to count of documents
        """
        distribution = defaultdict(int)
        for doc_id, core_class in self.core_classes.items():
            distribution[core_class] += 1
        
        return dict(distribution)
    
    def filter_low_confidence_docs(self, min_confidence: float = 0.0) -> Dict[int, int]:
        """
        Filter documents with low confidence scores
        
        Args:
            min_confidence: Minimum confidence threshold
        
        Returns:
            Filtered core_classes dictionary
        """
        filtered = {}
        for doc_id, core_class in self.core_classes.items():
            if self.confidence_scores[doc_id] >= min_confidence:
                filtered[doc_id] = core_class
        
        print(f"Filtered {len(filtered)}/{len(self.core_classes)} documents with confidence >= {min_confidence}")
        
        return filtered


class CoreClassAnalyzer:
    """Analyze and visualize core class mining results"""
    
    def __init__(self, core_miner: CoreClassMiner, hierarchy):
        """
        Initialize analyzer
        
        Args:
            core_miner: CoreClassMiner object
            hierarchy: TaxonomyHierarchy object
        """
        self.core_miner = core_miner
        self.hierarchy = hierarchy
    
    def get_level_distribution(self) -> Dict[int, int]:
        """
        Get distribution of core classes across hierarchy levels
        
        Returns:
            Dictionary mapping level to count
        """
        level_dist = defaultdict(int)
        
        for doc_id, core_class in self.core_miner.core_classes.items():
            level = self.hierarchy.get_level(core_class)
            level_dist[level] += 1
        
        return dict(sorted(level_dist.items()))
    
    def get_top_core_classes(self, k: int = 20) -> List[Tuple[int, str, int]]:
        """
        Get top-k most frequent core classes
        
        Args:
            k: Number of top classes to return
        
        Returns:
            List of (class_id, class_name, count) tuples
        """
        distribution = self.core_miner.analyze_core_class_distribution()
        
        # Sort by count
        sorted_classes = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        
        # Get top-k with names
        top_classes = []
        for class_id, count in sorted_classes[:k]:
            class_name = self.hierarchy.id_to_name.get(class_id, "Unknown")
            top_classes.append((class_id, class_name, count))
        
        return top_classes
    
    def print_summary(self):
        """Print summary of core class mining"""
        stats = self.core_miner.get_statistics()
        
        print("\n" + "="*60)
        print("Core Class Mining Summary")
        print("="*60)
        print(f"Documents with core class: {stats['num_docs_with_core_class']}")
        print(f"Avg candidates per doc: {stats['avg_candidates_per_doc']:.2f}")
        print(f"Avg confidence score: {stats['avg_confidence_score']:.4f}")
        print(f"Min confidence score: {stats['min_confidence_score']:.4f}")
        print(f"Max confidence score: {stats['max_confidence_score']:.4f}")
        
        print("\nLevel Distribution:")
        level_dist = self.get_level_distribution()
        for level, count in level_dist.items():
            print(f"  Level {level}: {count} documents")
        
        print("\nTop-10 Core Classes:")
        top_classes = self.get_top_core_classes(10)
        for class_id, class_name, count in top_classes:
            level = self.hierarchy.get_level(class_id)
            print(f"  {class_name} (ID: {class_id}, Level: {level}): {count} documents")
        
        print("="*60 + "\n")

