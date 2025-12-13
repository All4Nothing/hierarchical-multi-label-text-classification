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
    def __init__(self, encoder_name, enc_dim, class_embeddings, temperature=0.07):
        super(ClassModel, self).__init__()
        self.doc_encoder = AutoModel.from_pretrained(encoder_name)
        self.doc_dim = enc_dim

        self.temperature = temperature
        
        self.num_classes, self.label_dim = class_embeddings.size()
        # [중요] BERT 공간에 맞춰진 임베딩을 파라미터로 등록
        self.label_embedding_weights = nn.Parameter(class_embeddings.clone(), requires_grad=True)
        
    def mean_pooling(self, model_output, attention_mask):
        """
        MPNet(SBERT) Standard Pooling Strategy
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, input_ids, attention_mask):
        # 1. Document Encoding (MPNet)
        outputs = self.doc_encoder(input_ids, attention_mask=attention_mask)
        
        # 2. Mean Pooling (Crucial for MPNet)
        doc_vector = self.mean_pooling(outputs, attention_mask) # [Batch, 768]
        
        # 3. Normalization (Crucial for preventing Logit Explosion)
        # MPNet은 Cosine Similarity 공간에서 학습되었으므로 정규화가 필수입니다.
        doc_norm = F.normalize(doc_vector, p=2, dim=1)
        label_norm = F.normalize(self.label_embedding_weights, p=2, dim=1)
        
        # 4. Simple Dot Product (Cosine Similarity)
        # LBM 없이 직접 내적합니다.
        # 수식: Score = (Doc / |Doc|) * (Label / |Label|)^T
        scores = torch.matmul(doc_norm, label_norm.T) # [Batch, Num_Classes]
        
        # 5. Temperature Scaling
        # Cosine Sim(-1~1)을 Sigmoid에 적합한 범위(예: -14~14)로 확장
        scores = scores / self.temperature
        
        return scores

def multilabel_bce_loss_w(output, target, weight=None):
    if weight is None:
        weight = torch.ones_like(output)
    # reduction='sum' matches authors' code, normalizing by batch size manually if needed
    loss = F.binary_cross_entropy_with_logits(output, target, weight, reduction="sum")
    return loss / output.size(0)


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
    def __init__(self, texts, tokenizer, hierarchy_graph, num_classes, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.hierarchy_graph = hierarchy_graph
        self.num_classes = num_classes
        self.max_length = max_length
        
        # Precompute descendants (Required to avoid KeyError 0)
        self.descendants_cache = {}
        for node in range(num_classes):
            # Safe access for graph nodes
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

def select_top_down_beam(probs, graph, beam_width=10, min_labels=2, max_labels=3):
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
    def __init__(self, model_dir, hierarchy_graph, num_classes, device_id=0):
        self.device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
        self.hierarchy_graph = hierarchy_graph
        self.num_classes = num_classes
        
        model_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
            
        logger.info(f"Loading model architecture and weights from {model_dir}...")
        
        # 1. Initialize Structure with Dummy Embeddings 
        # (Real weights will be loaded via load_state_dict)
        dummy_embeddings = torch.zeros(num_classes, 768)

        target_model_name = "sentence-transformers/all-mpnet-base-v2"
        self.model = ClassModel(target_model_name, 768, dummy_embeddings, temperature=0.07)
        
        # 2. Load Trained Weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except:
            logger.warning("Tokenizer not found in model_dir, loading from huggingface hub...")
            self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)

    def predict(self, texts, batch_size=64, method="top_down_beam"):
        dataset = WeightedMultiLabelDataset(
            texts, self.tokenizer, self.hierarchy_graph, self.num_classes
        )
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        all_preds = []
        logger.info(f"Starting Inference with Method {method}...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(outputs)
                
                
                # preds = (probs > threshold).cpu().numpy()
                
                """for i, p in enumerate(preds):
                    indices = np.where(p)[0].tolist()
                    # Fallback: if no label predicted, pick top-1
                    if not indices:
                        indices = [torch.argmax(probs[i]).item()]
                    all_preds.append(indices)"""
                for doc_probs in probs:
                    # Apply Top-Down Beam Search
                    if method == "top_down_beam":
                        path = select_top_down_beam(
                            doc_probs, 
                            self.hierarchy_graph, 
                            beam_width=10,
                            min_labels=2, 
                            max_labels=3
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
    DATA_DIR = "../Amazon_products"  # Adjust if needed
    MODEL_DIR = "outputs/models/checkpoint_epoch_5"
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
        hierarchy_graph=loader.hierarchy_graph, # [FIX] Graph passed correctly
        num_classes=len(loader.all_classes)
    )
    
    # 3. Predict
    final_preds = engine.predict(loader.test_corpus, batch_size=128, method="top_down_beam")
    
    # 5. Save
    engine.save_submission(final_preds, loader.test_ids, OUTPUT_FILE)
    logger.info("Done!")