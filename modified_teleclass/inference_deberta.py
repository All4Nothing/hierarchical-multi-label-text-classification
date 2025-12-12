"""
TELEClass Inference-Only Script (DeBERTa Compatible)
Loads a trained model and generates Kaggle submission.
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
                
        # Load Hierarchy
        with open(os.path.join(self.data_dir, "class_hierarchy.txt"), 'r', encoding='utf-8') as f:
            for line in f:
                p, c = line.strip().split('\t')
                self.hierarchy_graph.add_edge(int(p), int(c))
        
        logger.info(f"Loaded {len(self.test_corpus)} test documents and {len(self.all_classes)} classes.")

class HierarchyExpander:
    def __init__(self, graph):
        self.graph = graph
        self.ancestors_cache = {}
        
    def get_ancestors(self, node):
        if node in self.ancestors_cache: return self.ancestors_cache[node]
        try:
            ancestors = nx.ancestors(self.graph, node)
        except:
            ancestors = set()
        self.ancestors_cache[node] = ancestors
        return ancestors

    def expand(self, labels_list):
        expanded = []
        for labels in tqdm(labels_list, desc="Expanding Hierarchy"):
            label_set = set(labels)
            for l in labels:
                label_set.update(self.get_ancestors(l))
            expanded.append(sorted(list(label_set)))
        return expanded

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
# 3. PATH SEARCH LOGIC
# ============================================================================

def select_top_down_beam(probs, graph, beam_width=5, min_labels=2, max_labels=3):
    """Top-Down Beam Search Strategy"""
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()

    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    
    beam = []
    for r in roots:
        if r < len(probs):
            score = probs[r]
            beam.append((score, [r]))
    
    beam = sorted(beam, key=lambda x: x[0], reverse=True)[:beam_width]
    completed_paths = [] 

    for _ in range(max_labels - 1):
        candidates = []
        for score, path in beam:
            curr_node = path[-1]
            if min_labels <= len(path) <= max_labels:
                completed_paths.append((score, path))
            
            children = list(graph.successors(curr_node))
            if not children: continue
                
            for child in children:
                if child >= len(probs): continue
                new_score = score * probs[child]
                new_path = path + [child]
                candidates.append((new_score, new_path))
        
        if not candidates: break
        beam = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
    
    completed_paths.extend(beam)
    
    valid_paths = [(s, p) for s, p in completed_paths if min_labels <= len(p) <= max_labels]
    
    if not valid_paths:
        return [int(np.argmax(probs))]
    
    alpha = 2 # larger alpha -> more weight on length
    best_path = sorted(valid_paths, key=lambda x: math.pow(x[0], 1.0 / (len(x[1]) ** alpha)), reverse=True)[0][1]
    return sorted(best_path)

def get_paths_to_root(graph, node):
    """
    Find all paths from Root to the given Node.
    Returns: List[List[int]] (e.g., [[root, parent, node]])
    """
    paths = []
    try:
        # NetworkX predecessors logic (Reverse DFS)
        preds = list(graph.predecessors(node))
        if not preds: # Root or isolated
            return [[node]]
        
        for p in preds:
            parent_paths = get_paths_to_root(graph, p)
            for path in parent_paths:
                paths.append(path + [node])
    except nx.NetworkXError:
        return [[node]]
    return paths

def select_leaf_priority(probs, graph, min_labels=2, max_labels=3):
    """
    Leaf-Priority Search (Strict Tree Version)
    - 가정: 모든 노드는 최대 1개의 부모만 가진다 (Tree 구조).
    - 로직: 가장 확률 높은 Leaf를 선택하고, 외길인 부모를 타고 올라간다.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()

    # 1. Leaf 노드 식별 (자식이 없는 노드)
    # 전체 노드 탐색 비용을 줄이기 위해 캐싱하면 좋으나, 여기서는 안전하게 매번 계산
    leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]
    
    # 유효한 인덱스만 필터링 (probs 범위 내)
    leaves = [l for l in leaves if l < len(probs)]
    
    if not leaves:
        # Fallback: 리프가 없으면(고립 노드 등) 전체 중 1등 반환
        return [int(np.argmax(probs))]

    # 2. Best Leaf 선택 (가장 강력한 신호)
    leaf_probs = probs[leaves]
    best_leaf = leaves[np.argmax(leaf_probs)]
    
    # 3. 경로 재구성 (Bottom-up Traversal)
    # 트리 구조이므로 부모는 무조건 0개(루트) 아니면 1개입니다.
    path = [best_leaf]
    curr = best_leaf
    
    while True:
        try:
            preds = list(graph.predecessors(curr))
            if not preds:
                # print(f"!!!! no predecessors: {path}")
                break # 루트 도달 (부모 없음)
            
            # [Tree Constraint] 부모는 오직 하나
            parent = preds[0]
            
            # Cycle 방지 (데이터 무결성 보호)
            if parent in path:
                print(f"!!!! cycle detected: {path}")
                break
                
            path.append(parent)
            curr = parent
            
            # [Safety Check] 계층 깊이가 3이므로, 경로가 3에 도달하면 즉시 중단
            # (만약 4개가 되면 데이터 오류지만, 코드는 안전하게 3개까지만 취함)
            if len(path) > max_labels:
                print(f"!!!! path over depth: {path}")
                break
        except:
            break
            
    # 현재 순서: [Leaf, Parent, Root] -> 뒤집어서 [Root, Parent, Leaf]
    final_path = path[::-1]
    
    # 4. 최소 길이 보정 (Padding)
    # 만약 Root 노드 하나만 예측되어 길이가 2 미만인 경우 (min_labels=2)
    if len(final_path) < min_labels:
        print(f"!!!! final_path under min_labels: {final_path}")
        sorted_indices = np.argsort(probs)[::-1]
        for idx in sorted_indices:
            idx = int(idx)
            if idx not in final_path:
                final_path.append(idx)
                if len(final_path) >= min_labels:
                    break
    
    return sorted(final_path)

def select_optimal_path(probs, graph, top_k=10, min_labels=2, max_labels=3):
    """
    Selects the best path chain with length [min_labels, max_labels]
    based on average probability.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()

    # 1. Select Candidate Anchors (Top-K Probabilities)
    sorted_indices = np.argsort(probs)[::-1] # Descending
    anchors = sorted_indices[:top_k]
    
    best_path = []
    best_score = -1.0
    
    for anchor in anchors:
        anchor = int(anchor)
        if not graph.has_node(anchor): continue
        
        # Get all paths from root to this anchor
        # e.g. [Root, L1, L2, Anchor]
        full_paths = get_paths_to_root(graph, anchor)
        
        for path in full_paths:
            # Generate all sub-paths (windows) of length [min_labels, max_labels]
            candidates = []
            
            # If path is short, take it all
            if len(path) < min_labels:
                candidates.append(path)
            else:
                # Sliding Window
                for length in range(min_labels, max_labels + 1):
                    if length > len(path): break
                    for i in range(len(path) - length + 1):
                        candidates.append(path[i : i + length])
            
            # Score each candidate
            for cand in candidates:
                # Score = Mean Probability
                score = np.mean([probs[c] for c in cand])
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_path = cand
    
    # Safety Fallback: Ensure minimum length
    final_path = sorted(list(set(best_path)))
    if len(final_path) < min_labels:
        # Add highest probability nodes that are not in path
        for idx in sorted_indices:
            idx = int(idx)
            if idx not in final_path:
                final_path.append(idx)
                if len(final_path) >= min_labels:
                    break
    
    return sorted(final_path)

# ============================================================================
# 4. INFERENCE ENGINE
# ============================================================================

class InferenceEngine:
    def __init__(self, model_dir, hierarchy_graph, num_classes, device_id=0):
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


        # [FIX] 2. Initialize with dynamic dimension
        dummy_embeddings = torch.zeros(num_classes, hidden_dim)
        self.model = ClassModel(config, hidden_dim, dummy_embeddings) # model_dir acts as encoder_name
        
        # 3. Load Trained Weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")

    def predict(self, texts, batch_size=64, method="top_down_beam"):
        dataset = WeightedMultiLabelDataset(
            texts, self.tokenizer, self.hierarchy_graph, self.num_classes
        )
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        all_preds = []
        logger.info(f"Starting Inference with Top-Down Beam Search...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                for doc_probs in probs:
                    # Apply Top-Down Beam Search
                    if method == "top_down_beam":
                        path = select_top_down_beam(
                            doc_probs, self.hierarchy_graph, beam_width=10,
                            min_labels=2, max_labels=3
                        )
                    elif method == "leaf_priority":
                        path = select_leaf_priority(
                            doc_probs, self.hierarchy_graph, min_labels=2, max_labels=3
                        )
                    elif method == "optimal_path":
                        path = select_optimal_path(
                            doc_probs, self.hierarchy_graph, top_k=10, min_labels=2, max_labels=3
                        )
                    else:
                        raise ValueError(f"Invalid method: {method}")
                    all_preds.append(path)
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
        
    method = "top_down_beam" # top_down_beam, leaf_priority, optimal_path
    MODEL_DIR = "outputs/models/deberta_model"
    OUTPUT_FILE = f"outputs/submission_deberta_{method}.csv"
    
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory 'Amazon_products' not found.")

    # 1. Load Data
    loader = DataLoaderSimple(DATA_DIR)
    loader.load()
    
    # 2. Setup Inference
    engine = InferenceEngine(
        model_dir=MODEL_DIR,
        hierarchy_graph=loader.hierarchy_graph,
        num_classes=len(loader.all_classes)
    )
    
    # 3. Predict & Optimize
    final_preds = engine.predict(loader.test_corpus, batch_size=128, method=method)
    
    
    # 4. Save
    engine.save_submission(final_preds, loader.test_ids, OUTPUT_FILE)
    logger.info("Done!")