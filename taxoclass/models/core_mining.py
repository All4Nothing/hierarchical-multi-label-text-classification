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
        self.core_classes = {}  # doc_id -> list of core class ids (multi-label)
        self.confidence_scores = {}  # doc_id -> dict of {class_id: confidence_score}
        
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
    
    def identify_core_classes(self) -> Dict[int, List[int]]:
        """
        Identify core classes for each document (multi-label)
        
        Paper: A class is a core class if its confidence score exceeds the threshold.
        Multiple classes can be core classes for one document.
        
        Returns:
            Dictionary mapping doc_id to list of core_class_ids
        """
        print("Step 1: Selecting candidate classes using top-down search...")
        for doc_id in tqdm(range(self.num_docs)):
            candidates = self.select_candidates_top_down(doc_id)
            self.candidate_classes[doc_id] = candidates
        
        print("Step 2: Computing confidence thresholds...")
        thresholds = self.compute_confidence_thresholds()
        
        print("Step 3: Identifying core classes (multi-label)...")
        for doc_id in tqdm(range(self.num_docs)):
            candidates = self.candidate_classes[doc_id]
            
            # Compute confidence scores for all candidates
            doc_core_classes = []
            doc_confidence_scores = {}
            
            for class_id in candidates:
                conf_score = self.compute_confidence_score(doc_id, class_id)
                threshold = thresholds.get(class_id, 0.0)
                
                # Check if confidence exceeds threshold
                # All classes that exceed threshold are core classes (multi-label)
                if conf_score >= threshold:
                    doc_core_classes.append(class_id)
                    doc_confidence_scores[class_id] = conf_score
            
            # Store multiple core classes
            if doc_core_classes:
                self.core_classes[doc_id] = doc_core_classes
                self.confidence_scores[doc_id] = doc_confidence_scores
            else:
                # Fallback: select class with highest similarity among candidates
                if candidates:
                    doc_similarities = self.similarity_matrix[doc_id]
                    best_class = max(candidates, key=lambda c: doc_similarities[c])
                    self.core_classes[doc_id] = [best_class]  # Still a list
                    self.confidence_scores[doc_id] = {best_class: doc_similarities[best_class]}
        
        total_core_classes = sum(len(cores) for cores in self.core_classes.values())
        avg_core_classes = total_core_classes / len(self.core_classes) if self.core_classes else 0
        print(f"Identified core classes for {len(self.core_classes)} documents")
        print(f"Total core classes: {total_core_classes}, Avg per doc: {avg_core_classes:.2f}")
        
        return self.core_classes
    
    def get_core_classes(self, doc_id: int) -> List[int]:
        """Get core classes for a document (returns list)"""
        return self.core_classes.get(doc_id, [])
    
    def get_confidence_scores(self, doc_id: int) -> Dict[int, float]:
        """Get confidence scores for a document (returns dict of {class_id: score})"""
        return self.confidence_scores.get(doc_id, {})
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistics about core class mining"""
        # Collect all confidence scores across all documents and classes
        all_conf_scores = []
        for doc_scores in self.confidence_scores.values():
            if isinstance(doc_scores, dict):
                all_conf_scores.extend(doc_scores.values())
            else:
                # Fallback for old single-value format
                all_conf_scores.append(doc_scores)
        
        total_core_classes = sum(len(cores) for cores in self.core_classes.values())
        avg_core_classes_per_doc = total_core_classes / len(self.core_classes) if self.core_classes else 0
        
        stats = {
            'num_docs_with_core_class': len(self.core_classes),
            'total_core_classes': total_core_classes,
            'avg_core_classes_per_doc': avg_core_classes_per_doc,
            'avg_candidates_per_doc': np.mean([len(candidates) for candidates in self.candidate_classes.values()]),
            'avg_confidence_score': np.mean(all_conf_scores) if all_conf_scores else 0.0,
            'min_confidence_score': np.min(all_conf_scores) if all_conf_scores else 0.0,
            'max_confidence_score': np.max(all_conf_scores) if all_conf_scores else 0.0
        }
        
        return stats
    
    def analyze_core_class_distribution(self) -> Dict[int, int]:
        """
        Analyze distribution of core classes
        
        Returns:
            Dictionary mapping class_id to count of documents
        """
        distribution = defaultdict(int)
        for doc_id, core_classes in self.core_classes.items():
            # core_classes is now a list
            for core_class in core_classes:
                distribution[core_class] += 1
        
        return dict(distribution)
    
    def filter_low_confidence_docs(self, min_confidence: float = 0.0) -> Dict[int, List[int]]:
        """
        Filter documents with low confidence scores
        
        Args:
            min_confidence: Minimum confidence threshold
        
        Returns:
            Filtered core_classes dictionary
        """
        filtered = {}
        for doc_id, core_classes in self.core_classes.items():
            # Filter core classes by confidence
            filtered_classes = []
            doc_scores = self.confidence_scores[doc_id]
            
            for class_id in core_classes:
                if isinstance(doc_scores, dict):
                    score = doc_scores.get(class_id, 0.0)
                else:
                    # Fallback for old single-value format
                    score = doc_scores
                
                if score >= min_confidence:
                    filtered_classes.append(class_id)
            
            if filtered_classes:
                filtered[doc_id] = filtered_classes
        
        print(f"Filtered {len(filtered)}/{len(self.core_classes)} documents with confidence >= {min_confidence}")
        
        return filtered


def create_training_labels(
    core_classes_dict: Dict[int, List[int]],
    hierarchy,
    num_classes: int,
    num_docs: int = None
) -> np.ndarray:
    """
    Create training labels from core classes for Stage 3 classifier training
    
    Paper: 
    - Positive set: Core classes + their ancestors (parents)
    - Negative set: All other classes except descendants of core classes
    - Ignore set (-1): Descendants of core classes (children)
    
    Args:
        core_classes_dict: Dictionary mapping doc_id to list of core_class_ids
        hierarchy: TaxonomyHierarchy object
        num_classes: Total number of classes
        num_docs: Total number of documents (if None, uses max(doc_id) + 1)
    
    Returns:
        Label matrix (num_docs, num_classes) where:
            1 = positive (core class or ancestor)
            0 = negative (other classes)
           -1 = ignore (descendants of core classes)
    """
    # Determine number of documents
    if num_docs is None:
        if core_classes_dict:
            num_docs = max(core_classes_dict.keys()) + 1
        else:
            num_docs = 0
    
    labels = np.zeros((num_docs, num_classes), dtype=np.float32)
    
    print("Creating training labels from core classes...")
    for doc_id in tqdm(range(num_docs)):
        if doc_id not in core_classes_dict:
            # Documents without core classes get all zeros (all negative)
            continue
        
        core_classes = core_classes_dict[doc_id]
        
        positive_set = set()
        ignore_set = set()
        
        for core_class in core_classes:
            # Add core class to positive set
            positive_set.add(core_class)
            
            # Add all ancestors to positive set
            ancestors = hierarchy.get_ancestors(core_class)
            positive_set.update(ancestors)
            
            # Add all descendants to ignore set
            descendants = hierarchy.get_descendants(core_class)
            ignore_set.update(descendants)
        
        # Remove overlap: positive set takes precedence over ignore set
        ignore_set = ignore_set - positive_set
        
        # Set labels
        for class_id in positive_set:
            labels[doc_id, class_id] = 1.0
        
        for class_id in ignore_set:
            labels[doc_id, class_id] = -1.0
        
        # All other classes remain 0 (negative)
    
    # Print statistics
    num_positive = (labels == 1).sum()
    num_negative = (labels == 0).sum()
    num_ignore = (labels == -1).sum()
    total = labels.size
    
    print(f"\nLabel Statistics:")
    print(f"  Positive: {num_positive} ({100*num_positive/total:.2f}%)")
    print(f"  Negative: {num_negative} ({100*num_negative/total:.2f}%)")
    print(f"  Ignore: {num_ignore} ({100*num_ignore/total:.2f}%)")
    print(f"  Avg positive per doc: {num_positive/num_docs:.2f}")
    
    return labels


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
        
        for doc_id, core_classes in self.core_miner.core_classes.items():
            # core_classes is now a list
            for core_class in core_classes:
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
        print(f"Total core classes: {stats['total_core_classes']}")
        print(f"Avg core classes per doc: {stats['avg_core_classes_per_doc']:.2f}")
        print(f"Avg candidates per doc: {stats['avg_candidates_per_doc']:.2f}")
        print(f"Avg confidence score: {stats['avg_confidence_score']:.4f}")
        print(f"Min confidence score: {stats['min_confidence_score']:.4f}")
        print(f"Max confidence score: {stats['max_confidence_score']:.4f}")
        
        print("\nLevel Distribution:")
        level_dist = self.get_level_distribution()
        for level, count in level_dist.items():
            print(f"  Level {level}: {count} core class assignments")
        
        print("\nTop-10 Core Classes:")
        top_classes = self.get_top_core_classes(10)
        for class_id, class_name, count in top_classes:
            level = self.hierarchy.get_level(class_id)
            print(f"  {class_name} (ID: {class_id}, Level: {level}): {count} documents")
        
        print("="*60 + "\n")

