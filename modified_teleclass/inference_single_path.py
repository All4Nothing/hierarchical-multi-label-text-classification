"""
TELEClass Inference with Single Path Selection
Uses similarity-based parent selection to generate single paths from leaf to root.
Based on TopDownLabelGenerator._get_single_path_to_root() method.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, DebertaV2Tokenizer
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
import logging
from typing import List, Dict, Optional, Set, Tuple
from sentence_transformers import SentenceTransformer

os.environ['TOKENIZERS_PARALLELISM'] = 'false' 

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 1. MODEL ARCHITECTURE (Must match teleclass_deberta.py)
# ============================================================================

class LBM(nn.Module):
    def __init__(self, l_dim, r_dim, n_classes=None, bias=True):
        super(LBM, self).__init__()
        self.weight = Parameter(torch.Tensor(l_dim, r_dim))
        self.use_bias = bias
        if self.use_bias:
            self.bias = Parameter(torch.Tensor(n_classes))
        bound = 1.0 / math.sqrt(l_dim)
        init.uniform_(self.weight, -bound, bound)
        if self.use_bias:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, e1, e2):
        scores = torch.matmul(torch.matmul(e1, self.weight), e2.T)
        if self.use_bias:
            scores = scores + self.bias
        return scores

class ClassModel(nn.Module):
    def __init__(self, config_or_name, enc_dim, class_embeddings):
        super(ClassModel, self).__init__()

        if isinstance(config_or_name, str):
            self.doc_encoder = AutoModel.from_pretrained(config_or_name)
        else:
            self.doc_encoder = AutoModel.from_config(config_or_name)
        self.doc_dim = enc_dim
        
        self.num_classes, self.label_dim = class_embeddings.size()
        self.label_embedding_weights = nn.Parameter(class_embeddings.clone(), requires_grad=True)
        self.interaction = LBM(self.doc_dim, self.label_dim, n_classes=self.num_classes, bias=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.doc_encoder(input_ids, attention_mask=attention_mask)
        doc_tensor = outputs.last_hidden_state[:, 0, :]
        scores = self.interaction(doc_tensor, self.label_embedding_weights)
        return scores

# ============================================================================
# 2. UTILS & DATA LOADING
# ============================================================================

class DataLoaderSimple:
    """Minimal Data Loader for Inference"""
    def __init__(self, data_dir="Amazon_products"):
        self.data_dir = data_dir
        self.test_corpus = []
        self.test_ids = []
        self.hierarchy_graph = nx.DiGraph()
        self.idx_to_class = {}
        self.class_to_idx = {}
        self.all_classes = []

    def load(self):
        logger.info("Loading test data and metadata...")
        # Load Test
        with open(os.path.join(self.data_dir, "test", "test_corpus.txt"), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2: 
                    self.test_ids.append(parts[0])
                    self.test_corpus.append(parts[1])
        
        # Load Classes
        with open(os.path.join(self.data_dir, "classes.txt"), 'r', encoding='utf-8') as f:
            for line in f:
                cid, cname = line.strip().split('\t')
                self.all_classes.append(cname)
                self.idx_to_class[int(cid)] = cname
                self.class_to_idx[cname] = int(cid)
                
        # Load Hierarchy
        with open(os.path.join(self.data_dir, "class_hierarchy.txt"), 'r', encoding='utf-8') as f:
            for line in f:
                p, c = line.strip().split('\t')
                self.hierarchy_graph.add_edge(int(p), int(c))
        
        logger.info(f"Loaded {len(self.test_corpus)} test documents and {len(self.all_classes)} classes.")

class WeightedMultiLabelDataset(Dataset):
    def __init__(self, texts, tokenizer, hierarchy_graph, num_classes, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.hierarchy_graph = hierarchy_graph
        self.num_classes = num_classes
        self.max_length = max_length
        
        self.descendants_cache = {}
        for node in range(num_classes):
            if hierarchy_graph.has_node(node):
                self.descendants_cache[node] = nx.descendants(hierarchy_graph, node)
            else:
                self.descendants_cache[node] = set()

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

# ============================================================================
# 3. SINGLE PATH SELECTION (from generator.py)
# ============================================================================

class SinglePathSelector:
    """
    Selects single path from leaf to root using similarity-based parent selection.
    Based on TopDownLabelGenerator._get_single_path_to_root() method.
    """
    
    def __init__(
        self, 
        hierarchy_graph: nx.DiGraph,
        all_classes: List[str],
        class_to_idx: Dict[str, int],
        sbert_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        self.hierarchy_graph = hierarchy_graph
        self.all_classes = all_classes
        self.class_to_idx = class_to_idx
        
        # Find root and root children
        self.root_id = self._find_root()
        self.root_children = set(self.hierarchy_graph.successors(self.root_id))
        logger.info(f"Root ID: {self.root_id}, Root children: {sorted(self.root_children)}")
        
        # Find leaf nodes
        self.leaf_nodes = self._find_leaf_nodes()
        logger.info(f"Found {len(self.leaf_nodes)} leaf nodes")
        
        # Load SBERT for class embeddings
        logger.info(f"Loading SBERT model: {sbert_model_name}")
        self.sbert_model = SentenceTransformer(sbert_model_name)
        
        # Generate class embeddings
        self._generate_class_embeddings()
        
        # Storage for document embeddings
        self.doc_embeddings: Dict[int, np.ndarray] = {}
    
    def _find_root(self) -> int:
        """Find root node (node with no incoming edges)"""
        roots = [n for n in self.hierarchy_graph.nodes() if self.hierarchy_graph.in_degree(n) == 0]
        if len(roots) == 0:
            raise ValueError("No root node found in hierarchy")
        if len(roots) > 1:
            logger.warning(f"Multiple roots found: {roots}, using first one")
        return roots[0]
    
    def _find_leaf_nodes(self) -> Set[int]:
        """Find all leaf nodes (nodes with no children)"""
        leaves = set()
        for node in self.hierarchy_graph.nodes():
            if self.hierarchy_graph.out_degree(node) == 0:
                leaves.add(node)
        return leaves
    
    def _generate_class_embeddings(self):
        """Generate embeddings for all classes using SBERT"""
        logger.info("Generating class embeddings...")
        self.class_embeddings: Dict[int, np.ndarray] = {}
        
        # Prepare all class texts
        class_texts = []
        class_ids = []
        for class_id in self.hierarchy_graph.nodes():
            class_name = self.all_classes[class_id] if class_id < len(self.all_classes) else f"class_{class_id}"
            class_text = class_name.replace('_', ' ')
            class_texts.append(class_text)
            class_ids.append(class_id)
        
        # Batch encode all classes at once
        logger.info(f"Encoding {len(class_texts)} classes...")
        embeddings = self.sbert_model.encode(
            class_texts, 
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Store embeddings
        for class_id, emb in zip(class_ids, embeddings):
            self.class_embeddings[class_id] = emb
    
    def encode_documents(self, texts: List[str], doc_ids: List[int]):
        """Encode documents using SBERT in batch"""
        logger.info(f"Encoding {len(texts)} documents...")
        
        # Batch encode all documents at once
        embeddings = self.sbert_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=64
        )
        
        # Store embeddings with doc_ids
        for doc_id, emb in zip(doc_ids, embeddings):
            self.doc_embeddings[doc_id] = emb
    
    def _select_best_parent_by_similarity(
        self,
        doc_id: int,
        class_id: int
    ) -> Optional[int]:
        """
        Select the best parent for a class based on document similarity.
        Returns the parent_id with highest similarity to the document.
        """
        parents = list(self.hierarchy_graph.predecessors(class_id))
        
        if not parents:
            return None

        valid_parents = [p for p in parents if p not in self.leaf_nodes]
        if not valid_parents:
            return None
        
        if len(valid_parents) == 1:
            return valid_parents[0]
        
        # Get document embedding
        doc_emb = self.doc_embeddings.get(doc_id)
        if doc_emb is None:
            return valid_parents[0]
        
        # Normalize document embedding
        doc_emb_norm = doc_emb / (np.linalg.norm(doc_emb) + 1e-8)
        
        # Compute similarity with each parent
        best_parent_id = None
        best_similarity = -1.0
        
        for parent_id in valid_parents:
            parent_emb = self.class_embeddings.get(parent_id)
            if parent_emb is None:
                continue
            
            # Normalize parent embedding
            parent_emb_norm = parent_emb / (np.linalg.norm(parent_emb) + 1e-8)
            
            # Compute cosine similarity
            similarity = float(np.dot(doc_emb_norm, parent_emb_norm))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_parent_id = parent_id
        
        # Fallback if no valid parent embedding found
        if best_parent_id is None:
            best_parent_id = valid_parents[0]
        
        return best_parent_id
    
    def get_single_path_to_root(
        self, 
        class_id: int, 
        doc_id: int
    ) -> Set[int]:
        """
        Get a single complete path from leaf to domain (real root).
        At each multi-parent node, select the best parent based on similarity.
        Stops at domain level (root's direct children) and excludes virtual root.
        """
        path = [class_id]  # Start with leaf
        current_id = class_id
        visited = set([class_id])  # Prevent cycles
        
        while True:
            # Check if current node is a domain (direct child of virtual root)
            if current_id in self.root_children:
                break
            
            parents = list(self.hierarchy_graph.predecessors(current_id))
            
            # If no parents at all, treat current node as domain (top-level node)
            if not parents:
                # Current node has no parents, it's effectively a domain node
                break
            
            # Filter out leaf nodes and already visited nodes
            valid_parents = [
                p for p in parents 
                if p not in self.leaf_nodes and p not in visited
            ]
            
            if not valid_parents:
                # No valid parents - check if any parent exists that's not a leaf
                # This might mean the current node should be treated as domain
                all_non_leaf_parents = [p for p in parents if p not in self.leaf_nodes]
                if not all_non_leaf_parents:
                    # All parents are leaves (shouldn't happen), use current node as domain
                    break
                # If we're stuck due to visited nodes, treat current as effective domain
                break
            
            # Select best parent
            if len(valid_parents) == 1:
                best_parent = valid_parents[0]
            else:
                # Multi-parent - use similarity to select
                best_parent = self._select_best_parent_by_similarity(doc_id, current_id)
                
                # Fallback if similarity selection fails
                if best_parent is None or best_parent not in valid_parents:
                    best_parent = valid_parents[0]
            
            # Stop if we reached a domain node
            if best_parent in self.root_children:
                path.append(best_parent)
                break
            
            # Stop if we reached virtual root
            if best_parent == self.root_id:
                break
            
            path.append(best_parent)
            visited.add(best_parent)
            current_id = best_parent
        
        # If path length exceeds 3, remove nodes from the beginning
        while len(path) > 3:
            path = path[1:]
        
        # Don't warn about path length - just return what we have
        # Some nodes may not have complete paths to root due to hierarchy structure
        
        return set(path)

# ============================================================================
# 4. INFERENCE ENGINE
# ============================================================================

class InferenceEngine:
    def __init__(self, model_dir, hierarchy_graph, num_classes, all_classes, class_to_idx, device_id=0):
        self.device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
        self.hierarchy_graph = hierarchy_graph
        self.num_classes = num_classes
        
        model_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
            
        logger.info(f"Loading model architecture and weights from {model_dir}...")
        
        config = AutoConfig.from_pretrained(model_dir)
        hidden_dim = config.hidden_size
        logger.info(f"Detected Config Hidden Dim: {hidden_dim}")

        # Initialize model
        dummy_embeddings = torch.zeros(num_classes, hidden_dim)
        self.model = ClassModel(config, hidden_dim, dummy_embeddings)
        
        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")
        
        # Initialize single path selector
        self.path_selector = SinglePathSelector(
            hierarchy_graph=hierarchy_graph,
            all_classes=all_classes,
            class_to_idx=class_to_idx
        )

    def predict(self, texts, doc_ids, batch_size=64):
        """
        Predict labels using single path selection (generator.py style).
        
        Process:
        1. Select Top-1 leaf node for each document
        2. Generate single path from leaf to root using similarity-based parent selection
        3. Return the path as final labels (2-3 nodes: leaf + intermediate(s) + domain)
        
        Args:
            texts: List of document texts
            doc_ids: List of document IDs
            batch_size: Batch size for inference
        """
        # First, encode documents for similarity computation
        logger.info("Encoding documents for similarity computation...")
        self.path_selector.encode_documents(texts, doc_ids)
        
        # Create dataset
        dataset = WeightedMultiLabelDataset(
            texts, self.tokenizer, self.hierarchy_graph, self.num_classes
        )
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        all_preds = []
        logger.info("Starting inference with single path selection (Top-1 leaf)...")
        
        doc_idx = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                for doc_probs in probs:
                    doc_id = doc_ids[doc_idx]
                    doc_idx += 1
                    
                    # Step 1: Select Top-1 leaf node (highest probability among all leaf nodes)
                    leaf_probs = []
                    for leaf_id in self.path_selector.leaf_nodes:
                        if leaf_id < len(doc_probs):
                            leaf_probs.append((leaf_id, doc_probs[leaf_id]))
                    
                    if not leaf_probs:
                        # No leaf nodes found, use empty prediction
                        all_preds.append([])
                        continue
                    
                    # Sort by probability and select top-1
                    leaf_probs.sort(key=lambda x: x[1], reverse=True)
                    top1_leaf_id = leaf_probs[0][0]
                    
                    # Step 2: Generate single path from leaf to root
                    # This uses similarity-based parent selection for multi-parent nodes
                    path = self.path_selector.get_single_path_to_root(top1_leaf_id, doc_id)
                    
                    # Step 3: Convert path (set) to sorted list for submission
                    final_labels = sorted(list(path))
                    all_preds.append(final_labels)
        
        return all_preds

    def save_submission(self, predictions, ids, output_path):
        logger.info(f"Saving submission to {output_path}")
        data = []
        for doc_id, label_indices in zip(ids, predictions):
            label_str = ",".join(map(str, label_indices))
            data.append({'id': doc_id, 'labels': label_str})
            
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        avg_len = df['labels'].apply(lambda x: len(x.split(','))).mean()
        logger.info(f"Submission Stats: Average Labels per Doc = {avg_len:.2f}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Settings
    DATA_DIR = "Amazon_products" 
    if not os.path.exists(DATA_DIR) and os.path.exists("../Amazon_products"):
        DATA_DIR = "../Amazon_products"
        
    MODEL_DIR = "outputs/models/deberta_model"
    OUTPUT_FILE = "outputs/submission_deberta_single_path.csv"
    
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory 'Amazon_products' not found.")

    # 1. Load Data
    loader = DataLoaderSimple(DATA_DIR)
    loader.load()
    
    # 2. Setup Inference
    engine = InferenceEngine(
        model_dir=MODEL_DIR,
        hierarchy_graph=loader.hierarchy_graph,
        num_classes=len(loader.all_classes),
        all_classes=loader.all_classes,
        class_to_idx=loader.class_to_idx
    )
    
    # 3. Predict with single path selection (generator.py style)
    # - Selects Top-1 leaf node per document
    # - Generates single path from leaf to root using similarity-based parent selection
    # - Returns path as final labels (2-3 nodes)
    test_doc_ids = [int(doc_id) for doc_id in loader.test_ids]
    final_preds = engine.predict(
        loader.test_corpus, 
        test_doc_ids, 
        batch_size=128
    )
    
    # 4. Save
    engine.save_submission(final_preds, loader.test_ids, OUTPUT_FILE)
    logger.info("Done!")
