"""
Taxonomy Hierarchy Processing Utilities
"""
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class TaxonomyHierarchy:
    """Manages taxonomy hierarchy structure"""
    
    def __init__(self, hierarchy_file: str, classes_file: str):
        """
        Initialize taxonomy hierarchy
        
        Args:
            hierarchy_file: Path to class_hierarchy.txt
            classes_file: Path to classes.txt
        """
        self.hierarchy_file = hierarchy_file
        self.classes_file = classes_file
        
        # Load class information
        self.id_to_name, self.name_to_id = self._load_classes()
        self.num_classes = len(self.id_to_name)
        
        # Build hierarchy graph
        self.graph = self._build_graph()
        
        # Precompute relationships
        self.parent_map = self._build_parent_map()
        self.children_map = self._build_children_map()
        self.ancestors_map = self._build_ancestors_map()
        self.descendants_map = self._build_descendants_map()
        self.siblings_map = self._build_siblings_map()
        self.levels = self._compute_levels()
        
        print(f"Loaded taxonomy: {self.num_classes} classes, max depth: {max(self.levels.values())}")
    
    def _load_classes(self) -> Tuple[Dict[int, str], Dict[str, int]]:
        """Load class ID and name mappings"""
        id_to_name = {}
        name_to_id = {}
        
        with open(self.classes_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    class_id = int(parts[0])
                    class_name = parts[1]
                    id_to_name[class_id] = class_name
                    name_to_id[class_name] = class_id
        
        return id_to_name, name_to_id
    
    def _build_graph(self) -> nx.DiGraph:
        """Build directed graph from hierarchy file"""
        graph = nx.DiGraph()
        
        # Add all nodes
        for class_id in self.id_to_name.keys():
            graph.add_node(class_id)
        
        # Add edges (parent -> child)
        with open(self.hierarchy_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    parent_id = int(parts[0])
                    child_id = int(parts[1])
                    graph.add_edge(parent_id, child_id)
        
        return graph
    
    def _build_parent_map(self) -> Dict[int, List[int]]:
        """Map each node to its parents"""
        parent_map = defaultdict(list)
        for node in self.graph.nodes():
            parents = list(self.graph.predecessors(node))
            parent_map[node] = parents
        return dict(parent_map)
    
    def _build_children_map(self) -> Dict[int, List[int]]:
        """Map each node to its children"""
        children_map = defaultdict(list)
        for node in self.graph.nodes():
            children = list(self.graph.successors(node))
            children_map[node] = children
        return dict(children_map)
    
    def _build_ancestors_map(self) -> Dict[int, Set[int]]:
        """Map each node to all its ancestors"""
        ancestors_map = {}
        for node in self.graph.nodes():
            ancestors = set()
            current = node
            visited = {current}
            queue = list(self.parent_map.get(current, []))
            
            while queue:
                parent = queue.pop(0)
                if parent not in visited:
                    visited.add(parent)
                    ancestors.add(parent)
                    queue.extend(self.parent_map.get(parent, []))
            
            ancestors_map[node] = ancestors
        
        return ancestors_map
    
    def _build_descendants_map(self) -> Dict[int, Set[int]]:
        """Map each node to all its descendants"""
        descendants_map = {}
        for node in self.graph.nodes():
            descendants = set()
            queue = list(self.children_map.get(node, []))
            visited = set(queue)
            
            while queue:
                child = queue.pop(0)
                descendants.add(child)
                for grandchild in self.children_map.get(child, []):
                    if grandchild not in visited:
                        visited.add(grandchild)
                        queue.append(grandchild)
            
            descendants_map[node] = descendants
        
        return descendants_map
    
    def _build_siblings_map(self) -> Dict[int, Set[int]]:
        """Map each node to its siblings (nodes with same parent)"""
        siblings_map = {}
        for node in self.graph.nodes():
            siblings = set()
            for parent in self.parent_map.get(node, []):
                siblings.update(self.children_map.get(parent, []))
            siblings.discard(node)  # Remove self
            siblings_map[node] = siblings
        
        return siblings_map
    
    def _compute_levels(self) -> Dict[int, int]:
        """Compute depth level of each node from root"""
        levels = {}
        # Find root nodes (nodes with no parents)
        roots = [n for n in self.graph.nodes() if len(self.parent_map.get(n, [])) == 0]
        
        # BFS to compute levels
        queue = [(root, 0) for root in roots]
        visited = set(roots)
        
        while queue:
            node, level = queue.pop(0)
            levels[node] = level
            
            for child in self.children_map.get(node, []):
                if child not in visited:
                    visited.add(child)
                    queue.append((child, level + 1))
        
        return levels
    
    def get_parents(self, class_id: int) -> List[int]:
        """Get parent classes"""
        return self.parent_map.get(class_id, [])
    
    def get_children(self, class_id: int) -> List[int]:
        """Get child classes"""
        return self.children_map.get(class_id, [])
    
    def get_ancestors(self, class_id: int) -> Set[int]:
        """Get all ancestor classes"""
        return self.ancestors_map.get(class_id, set())
    
    def get_descendants(self, class_id: int) -> Set[int]:
        """Get all descendant classes"""
        return self.descendants_map.get(class_id, set())
    
    def get_siblings(self, class_id: int) -> Set[int]:
        """Get sibling classes"""
        return self.siblings_map.get(class_id, set())
    
    def get_level(self, class_id: int) -> int:
        """Get depth level of class"""
        return self.levels.get(class_id, -1)
    
    def get_edge_index(self) -> np.ndarray:
        """
        Get edge index for PyTorch Geometric
        Returns: (2, num_edges) array
        """
        edges = []
        for parent, child in self.graph.edges():
            edges.append([parent, child])
            edges.append([child, parent])  # Bidirectional
        
        if len(edges) == 0:
            return np.array([[], []], dtype=np.int64)
        
        return np.array(edges, dtype=np.int64).T
    
    def get_roots(self) -> List[int]:
        """Get root nodes"""
        return [n for n in self.graph.nodes() if len(self.parent_map.get(n, [])) == 0]
    
    def get_leaves(self) -> List[int]:
        """Get leaf nodes"""
        return [n for n in self.graph.nodes() if len(self.children_map.get(n, [])) == 0]
    
    def get_nodes_at_level(self, level: int) -> List[int]:
        """Get all nodes at a specific level"""
        return [node for node, node_level in self.levels.items() if node_level == level]
    
    def is_ancestor(self, ancestor_id: int, node_id: int) -> bool:
        """Check if ancestor_id is an ancestor of node_id"""
        return ancestor_id in self.ancestors_map.get(node_id, set())
    
    def is_descendant(self, descendant_id: int, node_id: int) -> bool:
        """Check if descendant_id is a descendant of node_id"""
        return descendant_id in self.descendants_map.get(node_id, set())

