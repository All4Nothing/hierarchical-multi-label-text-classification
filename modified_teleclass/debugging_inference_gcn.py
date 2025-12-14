"""
TELEClass Inference-Only Script
Loads a trained model and generates Kaggle submission.
"""

import math # Missing import fix
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Set, Optional

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
# 1. MODEL ARCHITECTURE (Must match training)
# ============================================================================

def build_normalized_adjacency(hierarchy_graph: nx.DiGraph, all_classes: List[str], idx_to_class_name: Dict[int, str]) -> torch.Tensor:
    """
    Constructs a symmetrically normalized adjacency matrix.
    Safe mapping: Raw ID -> Class Name -> Matrix Index (0 ~ N-1)
    """
    num_classes = len(all_classes)
    
    # 1. Map Class Name -> Matrix Index (0 ~ N-1)
    # all_classes 리스트의 순서가 곧 모델의 레이블 인덱스입니다.
    name_to_model_idx = {name: i for i, name in enumerate(all_classes)}
    
    # 2. Initialize with Identity (Self-loops)
    adj = torch.eye(num_classes)
    
    # 3. Add Edges with ID Mapping
    # hierarchy_graph의 노드는 Raw ID(int)로 되어 있다고 가정
    edge_count = 0
    for u_raw, v_raw in hierarchy_graph.edges():
        # Raw ID를 이용해 이름 조회
        u_name = idx_to_class_name.get(u_raw)
        v_name = idx_to_class_name.get(v_raw)
        
        # 두 노드가 모두 학습 대상 클래스 목록에 있을 때만 엣지 추가
        if u_name in name_to_model_idx and v_name in name_to_model_idx:
            u = name_to_model_idx[u_name]
            v = name_to_model_idx[v_name]
            
            # Symmetric Connection (Information flows both ways)
            adj[u, v] = 1.0
            adj[v, u] = 1.0
            edge_count += 1
            
    logger.info(f"Constructed Adjacency Matrix: {num_classes} nodes, {edge_count} edges (mapped from raw hierarchy)")

    # 4. Symmetric Normalization: D^-0.5 * A * D^-0.5
    row_sum = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    
    adj_normalized = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    
    return adj_normalized

# ============================================================================
# NEW: GNN LAYERS & MODEL
# ============================================================================

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_feat, adj):
        support = torch.mm(input_feat, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output

class GNNClassModel(nn.Module):
    def __init__(self, encoder_name, enc_dim, class_embeddings, adj_matrix, initial_temp=0.07):
        super(GNNClassModel, self).__init__()
        self.doc_encoder = AutoModel.from_pretrained(encoder_name)
        
        # Buffer로 등록하면 state_dict에 저장되고 GPU로 자동 이동됨 (업데이트는 안 됨)
        self.register_buffer('adj', adj_matrix)
        
        self.num_classes, self.label_dim = class_embeddings.size()
        
        # Initial Embeddings (MPNet output) -> Input Feature for GNN
        self.class_emb_input = nn.Parameter(class_embeddings.clone(), requires_grad=True)
        
        # GCN Layers (Dimension Preserving: 768 -> 768)
        self.gc1 = GraphConvolution(enc_dim, enc_dim)
        self.gc2 = GraphConvolution(enc_dim, enc_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)
        
        # Learnable Temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / initial_temp))

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        # A. Document Encoding
        doc_outputs = self.doc_encoder(input_ids, attention_mask=attention_mask)
        doc_vector = self.mean_pooling(doc_outputs, attention_mask)
        doc_norm = F.normalize(doc_vector, p=2, dim=1)
        
        # B. Label Encoding via GNN
        # Layer 1
        x = self.gc1(self.class_emb_input, self.adj)
        x = self.relu(x)
        x = self.dropout(x)
        x = x + self.class_emb_input # Residual 1
        
        # Layer 2
        x = self.gc2(x, self.adj)
        final_class_emb = x + self.class_emb_input # Residual 2
        
        label_norm = F.normalize(final_class_emb, p=2, dim=1)
        
        # C. Classification
        cosine_sim = torch.matmul(doc_norm, label_norm.T)
        scale = self.logit_scale.exp().clamp(max=100)
        
        return cosine_sim * scale


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
        with open(os.path.join(self.data_dir, "test", "test_corpus_debug.txt"), 'r', encoding='utf-8') as f:
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
            print(f"labels: {labels}\n")
            label_set = set(labels)
            for l in labels:
                label_set.update(self.get_ancestors(l))
            expanded.append(sorted(list(label_set)))
            print(f"expanded: {expanded}\n")
        return expanded

class WeightedMultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, hierarchy_graph, num_classes, 
                 is_augmented_mask=None, augmentation_scaling=1.0, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.hierarchy_graph = hierarchy_graph
        self.num_classes = num_classes
        self.max_length = max_length
        
        # [Logic from prepare_training_data.py]
        self.is_augmented_mask = is_augmented_mask # List[bool]
        self.augmentation_scaling = augmentation_scaling
        
        # Precompute descendants
        self.descendants_cache = {}
        for node in range(num_classes):
            self.descendants_cache[node] = nx.descendants(hierarchy_graph, node)

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label_indices = self.labels[idx]
        is_aug = self.is_augmented_mask[idx] if self.is_augmented_mask else False
        
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt'
        )
        
        target = torch.zeros(self.num_classes)
        target[label_indices] = 0.9
        
        # [Sample Mask Logic]
        # 1. Default mask is 1.0
        mask = torch.ones(self.num_classes)
        
        if is_aug:
            # [Author's Logic] Augmented data gets a uniform scaling weight
            # augmentation_scaling이 리스트인 경우 해당 샘플의 값을 사용
            scaling_value = self.augmentation_scaling[idx] if isinstance(self.augmentation_scaling, (list, tuple, np.ndarray)) else self.augmentation_scaling
            mask = mask * scaling_value
        else:
            # [Author's Logic] Real data: Mask out descendants of positive classes
            # (Ambiguous: If parent is true, child might be true or false, so don't penalize)
            mask[label_indices] *= 2.0 # 10.0 -> 2.0
            
            for pos_cls in label_indices:
                descendants = self.descendants_cache.get(pos_cls, set())
                for desc in descendants:
                    if desc not in label_indices:
                        mask[desc] = 0.0
            
            
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': target,
            'sample_mask': mask
        }



# ============================================================================
# 3. OPTIMAL PATH SEARCH LOGIC (Core Algorithm)
# ============================================================================

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

def select_top_down_beam(probs, graph, beam_width=10, min_labels=2, max_labels=3, alpha=3):
    """
    Top-Down Beam Search
    - 루트에서 시작하여 확률의 곱(Product)이 가장 높은 경로를 탐색
    - Score = P(Parent) * P(Child) * ...
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()

    # 1. 루트 노드 찾기 (들어오는 간선이 없는 노드)
    # 그래프에 따라 루트가 0번 하나일 수도, 여러 개일 수도 있음
    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    
    # Beam 초기화: (Score, Path) 튜플 리스트
    # Score 초기값은 1.0 (곱셈 항등원) 또는 해당 루트의 확률
    beam = []
    for r in roots:
        if r < len(probs): # 유효한 노드 인덱스인지 확인
            score = probs[r]
            beam.append((score, [r]))
    
    # 확률 높은 순 정렬 후 상위 K개만 유지
    beam = sorted(beam, key=lambda x: x[0], reverse=True)[:beam_width]
    
    completed_paths = [] # 완료된 경로 저장소

    # 2. 깊이 우선 탐색 (최대 깊이까지)
    # 이미 1단계(루트)는 했으므로 max_labels-1 번 더 반복
    for _ in range(max_labels - 1):
        candidates = []
        
        for score, path in beam:
            curr_node = path[-1]
            
            # 현재 경로 길이가 조건을 만족하면 완료 목록에도 후보로 등록
            if min_labels <= len(path) <= max_labels:
                completed_paths.append((score, path))
            
            # 자식 노드 확장
            children = list(graph.successors(curr_node))
            
            if not children: # 더 이상 자식이 없으면 종료
                continue
                
            for child in children:
                if child >= len(probs): continue
                
                # 점수 계산: 부모까지의 점수 * 자식 확률
                # (확률이 너무 작아지는 것을 방지하려면 log sum을 써도 됨)
                new_score = score * probs[child]
                new_path = path + [child]
                candidates.append((new_score, new_path))
        
        # 이번 단계에서 확장된 후보가 없으면 중단
        if not candidates:
            break
            
        # Beam Pruning: 상위 K개만 다음 단계로 가져감
        beam = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
    
    # 마지막 Beam에 남은 것들도 완료 목록에 추가
    completed_paths.extend(beam)
    
    # 3. 제약 조건 필터링 및 최적 경로 선택
    valid_paths = [
        (s, p) for s, p in completed_paths 
        if min_labels <= len(p) <= max_labels
    ]
    print(f"valid_paths: {valid_paths}\n")
    
    if not valid_paths:
        print(f"!!!! WARNING: no valid paths")
        # Fallback: 실패 시 가장 확률 높은 단일 노드라도 반환 (혹은 기존 방식 사용)
        best_idx = np.argmax(probs)
        return [int(best_idx)]
        
    # 점수가 가장 높은 경로 반환
    # alpha = 2 # larger alpha -> more weight on length
    # best_path = sorted(valid_paths, key=lambda x: math.pow(x[0], 1.0 / (len(x[1]) ** alpha)), reverse=True)[0][1]
    best_path = sorted(valid_paths, key=lambda x: x[0], reverse=True)[0][1]
    print(f"best_path: {best_path}\n")
    return sorted(best_path)

# ============================================================================
# 4. INFERENCE ENGINE
# ============================================================================

class InferenceEngine:
    def __init__(self, model_dir, hierarchy_graph, all_classes, idx_to_class, device_id=0):
        """
        GCN Inference Engine
        Args:
            all_classes: List of class names (defines the order of 0~N index)
            idx_to_class: Dict mapping Raw ID -> Class Name (for building graph)
        """
        self.device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
        self.hierarchy_graph = hierarchy_graph
        self.all_classes = all_classes
        self.num_classes = len(all_classes)
        
        model_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
            
        logger.info(f"Loading GCN model from {model_dir}...")

        # 1. Build Adjacency Matrix (Must match training logic!)
        # 학습 때와 똑같은 구조의 행렬을 만들어야 가중치가 제대로 작동합니다.
        adj_matrix = build_normalized_adjacency(
            hierarchy_graph, 
            all_classes, 
            idx_to_class
        )
        adj_matrix = adj_matrix.to(self.device)
        
        # 2. Initialize Structure with Dummy Embeddings 
        # (실제 가중치는 load_state_dict로 덮어씌워지므로 초기값은 0이어도 무관함)
        dummy_embeddings = torch.zeros(self.num_classes, 768)
        target_model_name = "sentence-transformers/all-mpnet-base-v2"
        
        # [MODIFIED] ClassModel -> GNNClassModel
        self.model = GNNClassModel(
            encoder_name=target_model_name, 
            enc_dim=768, 
            class_embeddings=dummy_embeddings, 
            adj_matrix=adj_matrix,
            initial_temp=0.07 # 불러올 때 덮어씌워지므로 초기값은 상관없음
        )
        
        # 3. Load Trained Weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except:
            logger.warning("Tokenizer not found in model_dir, loading from huggingface hub...")
            self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)

    def predict(self, texts, batch_size=64, method="top_down_beam", alpha=3):
        # [FIX] Dataset signature matching
        # Training Dataset expects: (texts, labels, tokenizer, graph, num_classes...)
        # We pass dummy labels (empty lists) for inference.
        dummy_labels = [[] for _ in range(len(texts))]
        
        dataset = WeightedMultiLabelDataset(
            texts, 
            dummy_labels,  # Dummy labels
            self.tokenizer, 
            self.hierarchy_graph, 
            self.num_classes,
            is_augmented_mask=None,
            augmentation_scaling=1.0
        )
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        all_preds = []
        doc_idx = 0
        logger.info(f"Starting Inference with Method {method}...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # GNN Forward Pass
                outputs = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(outputs)
                
                # --- Analysis Block (Optional) ---
                # Check specifics for debugging
                if doc_idx < 3: # Print first 3 docs only
                    doc_probs_np = probs[0].cpu().numpy() # First in batch
                    top_10_indices = np.argsort(doc_probs_np)[-10:][::-1]
                    top_10_values = doc_probs_np[top_10_indices]
                    
                    print(f"\n=== Document {doc_idx} Analysis ===")
                    print("Top 10 Classes:")
                    for idx, val in zip(top_10_indices, top_10_values):
                        cls_name = self.all_classes[idx] if idx < len(self.all_classes) else str(idx)
                        print(f"  {cls_name} ({idx}): {val:.6f}")
                # ---------------------------------

                # --- Search Strategy ---
                # 배치 처리를 위해 루프를 돌며 Search 적용
                # (Note: Search functions should be imported or defined)
                batch_probs = probs.cpu().numpy()
                for doc_probs in batch_probs:
                    # Get top 10 classes with their prediction values
                    top_10_indices = np.argsort(doc_probs_np)[-10:][::-1]  # Descending order
                    top_10_values = doc_probs_np[top_10_indices]
                    
                    # Get class 40 (baby_products) prediction value
                    class_40_value = doc_probs_np[40] if 40 < len(doc_probs_np) else 0.0
                    
                    # Print results for this document
                    print(f"\n=== Document {doc_idx} ===")
                    print("Top 10 Classes (Class Number: Prediction Value):")
                    for idx, val in zip(top_10_indices, top_10_values):
                        print(f"  Class {idx}: {val:.6f}")
                    print(f"Class 40 (baby_products): {class_40_value:.6f}")
                    
                    if method == "top_down_beam":
                        path = select_top_down_beam(
                            doc_probs, 
                            self.hierarchy_graph, 
                            beam_width=10,
                            min_labels=2, 
                            max_labels=3,
                            alpha=alpha
                        )
                    elif method == "leaf_priority":
                        path = select_leaf_priority(
                            doc_probs, self.hierarchy_graph, min_labels=2, max_labels=3
                        )
                    elif method == "optimal_path":
                        path = select_optimal_path(
                            doc_probs, self.hierarchy_graph, top_k=10, min_labels=2, max_labels=3
                        )
                    elif method == "threshold":
                         # Simple Threshold fallback
                         path = np.where(doc_probs > 0.15)[0].tolist()
                         if not path:
                             path = [np.argmax(doc_probs)]
                    else:
                        raise ValueError(f"Invalid method: {method}")
                    
                    all_preds.append(path)
                    doc_idx += 1
                    
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
    method = "top_down_beam" # top_down_beam, leaf_priority, optimal_path, 
    alpha=2 # larger alpha -> more weight on length
    DATA_DIR = "../Amazon_products"  # Adjust if needed
    MODEL_DIR = "outputs/models/checkpoint_epoch_1"
    OUTPUT_FILE = "outputs/submit_debugging.csv"
    
    if not os.path.exists(DATA_DIR):
        if os.path.exists(f"../{DATA_DIR}"): DATA_DIR = f"../{DATA_DIR}"
        else: raise FileNotFoundError("Data directory not found")

    # 1. Load Data
    loader = DataLoaderSimple(DATA_DIR)
    loader.load()
    
    # 2. Setup Inference
    engine = InferenceEngine(
        model_dir=MODEL_DIR,
        hierarchy_graph=loader.hierarchy_graph, 
        all_classes=loader.all_classes,      # [NEW] Pass class list
        idx_to_class=loader.idx_to_class,    # [NEW] Pass ID mapping
        device_id=0
    )
    
    # 3. Predict
    final_preds = engine.predict(loader.test_corpus, batch_size=64, method=method, alpha=alpha)
    
    # 5. Save
    engine.save_submission(final_preds, loader.test_ids, OUTPUT_FILE)
    logger.info("Done!")