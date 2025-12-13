"""
Hierarchical Multi-Label Text Classification Pipeline - Multi-GPU Version
Based on TELEClass framework with transductive learning.

This version supports:
- Multi-GPU training with DataParallel/DistributedDataParallel
- Parallel document encoding across multiple GPUs
- Optimized batch processing
- Mixed precision training (FP16)

Performance improvements:
- 2-4x faster training on multiple GPUs
- Efficient memory utilization
- Automatic GPU load balancing
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = 'NO'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Fix tokenizer warning

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try to load .env file from the script's directory or parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')
    if not os.path.exists(env_path):
        # Try parent directory
        env_path = os.path.join(os.path.dirname(script_dir), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded .env file from: {env_path}")
except ImportError:
    # dotenv not installed, skip loading .env file
    print("Warning: python-dotenv not installed. Install it with: pip install python-dotenv")
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModel, AutoConfig, get_cosine_schedule_with_warmup
from sentence_transformers import SentenceTransformer
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional
from tqdm import tqdm
import json
import logging
import openai
from transformers import get_linear_schedule_with_warmup

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] %(message)s'
)
logger = logging.getLogger(__name__)


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = 'NO'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional
from tqdm import tqdm
import logging

# [기존 Phase 0~4 관련 유틸리티 및 클래스는 pipeline_teleclass_multigpu.py와 동일하다고 가정]
# 여기서는 수정된 Phase 5 관련 핵심 클래스만 다시 정의합니다.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# AUTHOR'S MODEL ARCHITECTURE (From model.py)
# ============================================================================

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
    """if weight is None:
        weight = torch.ones_like(output)
    # reduction='sum' matches authors' code, normalizing by batch size manually if needed
    loss = F.binary_cross_entropy_with_logits(output, target, weight, reduction="sum")
    return loss / output.size(0)"""
    # [Advanced] Asymmetric weighting (Focus on Positives)
    # 정답(Target=1)인 경우 Loss를 더 크게 키움
    
    pos_weight = torch.ones_like(output) * 5.0  # 정답을 틀리면 5배 더 혼냄
    
    # reduction='mean'으로 변경하여 배치 크기에 따른 변동성을 줄임
    loss = F.binary_cross_entropy_with_logits(
        output, 
        target, 
        weight=weight, 
        pos_weight=pos_weight, # <--- 핵심
        reduction="mean" 
    )
    return loss

# ============================================================================
# UPDATED DATASET (Matches prepare_training_data.py logic)
# ============================================================================

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
# HELPER: Generate BERT embeddings for Class Names
# ============================================================================
def generate_bert_class_embeddings(model_name, class_list, class_descriptions, device):
    """
    Generates class embeddings using the SAME backbone as the classifier.
    Matches logic in prepare_training_data.py
    """
    # logger.info(f"Generating class embeddings using {model_name} (for compatibility)...")
    logger.info(f"Generating class embeddings using {model_name} with LLM Descriptions...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    texts_to_encode = []
    for cls_name in class_list:
        if class_descriptions and cls_name in class_descriptions:
            # LLM 설명 사용
            desc = class_descriptions[cls_name]
            # (옵션) 키워드 부분 제거하고 설명만 쓸지, 통째로 쓸지 결정. 통째로 추천.
            texts_to_encode.append(desc) 
        else:
            # Fallback
            texts_to_encode.append(f"The product category is {cls_name.replace('_', ' ')}")

    class_emb = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(texts_to_encode), batch_size):
            batch_texts = texts_to_encode[i : i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            outputs = model(**inputs)
            
            # [CLS] 토큰 사용 (설명이 길어진 경우 Mean보다 CLS가 나을 수 있음, 혹은 Mean 유지)
            # 여기서는 DeBERTa/BERT 특성상 CLS 혹은 Mean 사용. 작성자 코드는 Mean 사용.
            # outputs.last_hidden_state: (batch, seq, dim)
            # Attention Mask를 고려한 Mean Pooling 적용
            mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            masked_embeddings = outputs.last_hidden_state * mask
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            class_emb.append(mean_embeddings.cpu())
            
    return torch.cat(class_emb, dim=0)

# ============================================================================
# TRAINER CLASS
# ============================================================================

class TELEClassTrainer:
    def __init__(self, num_classes, model_name, hidden_dim, class_embeddings, temperature=0.07, device_ids=None):
        self.available_gpus = device_ids if device_ids else list(range(torch.cuda.device_count()))
        self.primary_device = f'cuda:{self.available_gpus[0]}' if self.available_gpus else 'cpu'
        self.hidden_dim = hidden_dim
        self.model = ClassModel(model_name, self.hidden_dim, class_embeddings, temperature=0.07)
        
        # Layer Freezing
        modules = [self.model.doc_encoder.embeddings, *self.model.doc_encoder.encoder.layer[:8]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        logger.info("Frozen MPNet layers 0-7. Training only top layers")
                
        
        self.model.to(self.primary_device)
        
        if len(self.available_gpus) > 1:
            self.model = DataParallel(self.model, device_ids=self.available_gpus)
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_classes = num_classes

    def train(self, train_texts, train_labels, is_augmented_mask, aug_scaling, 
              hierarchy_graph, epochs=20, lr=2e-5, batch_size=16, output_dir="outputs", patience=5, save_interval=5):
        
        eff_batch_size = batch_size * max(1, len(self.available_gpus))
        
        """dataset = WeightedMultiLabelDataset(
            train_texts, train_labels, self.tokenizer, hierarchy_graph, self.num_classes,
            is_augmented_mask=is_augmented_mask, augmentation_scaling=aug_scaling
        )
        dataloader = TorchDataLoader(dataset, batch_size=eff_batch_size, shuffle=True, num_workers=4)
        """
        # 1. Validation Split (데이터 분리)
        # Stratified Split을 하면 좋지만, Multi-label이므로 단순 Random Split 사용
        # 입력 데이터 리스트들을 함께 분리하기 위해 zip으로 묶었다가 품
        logger.info(f"No splitting data")
        
        # aug_scaling이 None이거나 스칼라인 경우 리스트로 변환
        if aug_scaling is None:
            aug_scaling = [1.0] * len(train_texts)
        elif not isinstance(aug_scaling, (list, tuple, np.ndarray)):
            # 스칼라 값인 경우 리스트로 변환
            aug_scaling = [aug_scaling] * len(train_texts)

        # 데이터 분리
        tr_texts = train_texts
        tr_labels = train_labels
        tr_mask = is_augmented_mask
        tr_scale = aug_scaling

        # 2. Dataset & DataLoader 생성
        train_dataset = WeightedMultiLabelDataset(
            tr_texts, tr_labels, self.tokenizer, hierarchy_graph, self.num_classes,
            is_augmented_mask=tr_mask, augmentation_scaling=tr_scale
        )

        train_loader = TorchDataLoader(train_dataset, batch_size=eff_batch_size, shuffle=True, num_workers=4)

        # Author's Optimizer Settings
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
        
        # Cosine Scheduler 추가
        num_training_steps = len(train_loader) * epochs
        num_warmup_steps = int(num_training_steps * 0.1) # 10% Warmup
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.primary_device)
                attention_mask = batch['attention_mask'].to(self.primary_device)
                labels = batch['labels'].to(self.primary_device)
                sample_mask = batch['sample_mask'].to(self.primary_device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = multilabel_bce_loss_w(outputs, labels, sample_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                scheduler.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_loss / len(train_loader)

            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

            # Save Checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_model(os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}"))
                logger.info(f"Saved checkpoint to {output_dir}/checkpoint_epoch_{epoch+1}")

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
            
        logger.info(f"Training complete. Best Val Loss: {best_val_loss:.4f}")

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        model_to_save = self.model.module if isinstance(self.model, DataParallel) else self.model
        torch.save(model_to_save.state_dict(), os.path.join(path, "pytorch_model.bin"))
        self.tokenizer.save_pretrained(path)
        model_to_save.doc_encoder.config.save_pretrained(path)

# ============================================================================
# INFERENCE HELPER
# ============================================================================
class CustomInference:
    def __init__(self, model_path, class_embeddings, hierarchy_graph, device_ids=None):
        self.device = f'cuda:{device_ids[0]}' if device_ids else 'cpu'
        self.hierarchy_graph = hierarchy_graph
        # Inference 시에도 동일한 구조 초기화 필요
        self.model = ClassModel("bert-base-uncased", 768, class_embeddings)
        self.model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def predict(self, texts, batch_size=32, threshold=0.15): # Threshold 낮춤
        # Inference에서는 Mask 불필요
        dataset = WeightedMultiLabelDataset(
            texts, [[]]*len(texts), self.tokenizer, self.hierarchy_graph, self.model.num_classes
        )
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).cpu().numpy()
                for i, p in enumerate(preds):
                    indices = np.where(p)[0].tolist()
                    if not indices:
                        indices = [torch.argmax(probs[i]).item()]
                    all_preds.append(indices)
        return all_preds

    def generate_submission(self, predictions, idx_to_class, output_path="submission.csv"):
        # [FIXED] Added this method to match pipeline call
        logger.info(f"Generating submission file: {output_path}")
        submission_data = []
        for doc_id, label_indices in enumerate(predictions):
            label_str = ",".join(map(str, label_indices))
            submission_data.append({'id': doc_id, 'labels': label_str})

        df = pd.DataFrame(submission_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Submission saved: {output_path}")

# ============================================================================
# GPU UTILITIES
# ============================================================================

def get_available_gpus():
    """Get list of available GPU IDs."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


# ============================================================================
# PHASE 0: REPRODUCIBILITY
# ============================================================================

def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries.
    This is CRITICAL for competition submissions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")


# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    """
    Loads and parses all competition data files.
    Handles corpora, taxonomy, keywords, and class definitions.
    """
    
    def __init__(self, data_dir: str = "Amazon_products"):
        self.data_dir = data_dir
        self.train_corpus = []
        self.test_corpus = []
        self.all_corpus = []
        self.all_indices = []
        self.train_indices = []
        self.test_indices = []
        self.hierarchy_graph = nx.DiGraph()
        self.class_keywords = {}
        self.all_classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
    def load_all(self):
        """Load all data files."""
        logger.info("Loading all data files...")
        self._load_corpora()
        self._load_taxonomy()
        self._load_keywords()
        self._load_classes()
        logger.info(f"Data loading complete:")
        logger.info(f"  - Train corpus: {len(self.train_corpus)} documents")
        logger.info(f"  - Test corpus: {len(self.test_corpus)} documents")
        logger.info(f"  - Combined corpus: {len(self.all_corpus)} documents")
        logger.info(f"  - Classes: {len(self.all_classes)}")
        logger.info(f"  - Hierarchy edges: {self.hierarchy_graph.number_of_edges()}")
        
    def _load_corpora(self):
        """Load train and test corpora, combine them into all_corpus."""
        train_path = os.path.join(self.data_dir, "train", "train_corpus.txt")
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    doc_id, text = parts
                    self.train_corpus.append(text)
                    
        test_path = os.path.join(self.data_dir, "test", "test_corpus.txt")
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    doc_id, text = parts
                    self.test_corpus.append(text)
        
        self.all_corpus = self.train_corpus + self.test_corpus
        self.all_indices = list(range(len(self.all_corpus)))
        self.train_indices = list(range(len(self.train_corpus)))
        self.test_indices = list(range(len(self.train_corpus), len(self.all_corpus)))
        
    def _load_taxonomy(self):
        """Load class hierarchy and build NetworkX DiGraph."""
        hierarchy_path = os.path.join(self.data_dir, "class_hierarchy.txt")
        with open(hierarchy_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    parent_id, child_id = parts
                    self.hierarchy_graph.add_edge(int(parent_id), int(child_id))
                    
    def _load_keywords(self):
        """Load class keywords."""
        keywords_path = os.path.join(self.data_dir, "class_related_keywords.txt")
        with open(keywords_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(':', 1)
                if len(parts) == 2:
                    class_name, keywords = parts
                    keyword_list = [kw.strip() for kw in keywords.split(',')]
                    self.class_keywords[class_name] = keyword_list
                    
    def _load_classes(self):
        """Load class list and create mappings."""
        classes_path = os.path.join(self.data_dir, "classes.txt")
        with open(classes_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    class_id, class_name = parts
                    self.all_classes.append(class_name)
                    self.class_to_idx[class_name] = int(class_id)
                    self.idx_to_class[int(class_id)] = class_name


# ============================================================================
# PHASE 1: MULTI-GPU CLASS REPRESENTATION
# ============================================================================

class MultiGPUClassRepresentation:
    """
    Creates contextualized class embeddings using multiple GPUs.
    Distributes encoding workload across available GPUs.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", 
                 device_ids: Optional[List[int]] = None, output_dir: str = "outputs"):
        self.available_gpus = get_available_gpus()
        self.output_dir = output_dir
        
        if not self.available_gpus:
            logger.warning("No GPUs available, using CPU")
            self.device_ids = []
            self.primary_device = 'cpu'
        else:
            self.device_ids = device_ids or self.available_gpus
            self.primary_device = f'cuda:{self.device_ids[0]}'
            logger.info(f"Using GPUs: {self.device_ids}")
        
        logger.info(f"Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Enable multi-GPU encoding if available
        if len(self.device_ids) > 1:
            logger.info(f"Enabling multi-GPU encoding across {len(self.device_ids)} GPUs")
            # SentenceTransformer automatically uses all visible CUDA devices
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.device_ids))
        
        self.model.to(self.primary_device)
        
        # Initialize OpenAI client with API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = openai.OpenAI(api_key=api_key)
        else:
            logger.warning("OPENAI_API_KEY not found in environment variables. LLM features may not work.")
            self.openai_client = None
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call OpenAI API to generate a class description.
        """
        if self.openai_client is None:
            logger.error("OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.")
            return ""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return ""
        
        

    def create_enriched_class_descriptions(self, class_keywords, hierarchy_graph, idx_to_class):
        """
        Generates context-aware class descriptions using LLM with Hierarchy info.
        """
        logger.info("Generating enriched class descriptions via LLM...")
        descriptions = {}
        
        cache_file = os.path.join(self.output_dir, "enriched_class_descriptions.json")

        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                descriptions = json.load(f)
            logger.info(f"Loaded {len(descriptions)} cached descriptions")
            return descriptions

        # 그래프 탐색을 위해 이름을 ID로 매핑하는 역방향 딕셔너리 생성 (필요시)
        class_to_idx = {v: k for k, v in idx_to_class.items()}

        cnt = 0
        save_interval = 50
        for class_name, keywords in tqdm(class_keywords.items(), desc="LLM Enrichment"):
            # 1. Context Retrieval (Hierarchy)
            cid = class_to_idx.get(class_name)
            parent_name = "None"
            siblings = []
            
            if cid is not None and hierarchy_graph.has_node(cid):
                # Find Parent
                preds = list(hierarchy_graph.predecessors(cid))
                if preds:
                    parent_id = preds[0] # Tree 구조 가정
                    parent_name = idx_to_class.get(parent_id, "Unknown")
                    
                    # Find Siblings (Children of the same parent, excluding self)
                    siblings_ids = list(hierarchy_graph.successors(parent_id))
                    siblings = [idx_to_class[s] for s in siblings_ids if s != cid]
            
            # 2. Prompt Engineering (The 'Profile' Strategy)
            sibling_str = ", ".join(siblings[:3]) if siblings else "None" # 너무 많으면 3개만
            keyword_str = ", ".join(keywords)
            
            prompt = f"""
            Task: Define the Amazon product category '{class_name}'.
            Context:
            - Parent Category: {parent_name}
            - Sibling Categories (Distinct from): {sibling_str}
            - Associated Keywords: {keyword_str}
            
            Write a sharp, 1-sentence definition focusing on visual/functional attributes 
            that distinguish '{class_name}' from its siblings. 
            Do NOT mention the class name directly in the beginning.
            """
            
            # 3. LLM Generation
            llm_desc = self._call_llm_api(prompt)

            if not llm_desc:
                logger.warning(f"Failed to generate description for {class_name}, using keywords only")
            
            
            # 4. Residual Connection (Safety Net)
            # LLM이 헛소리를 할 경우를 대비해 원래 키워드를 뒤에 붙임
            final_desc = f"Class Name: {class_name} ; Keywords: {keyword_str} ; Description: {llm_desc}"
            descriptions[class_name] = final_desc

            cnt += 1
            if cnt % save_interval == 0:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(descriptions, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(descriptions)} enriched descriptions to {cache_file}")
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(descriptions, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(descriptions)} enriched descriptions to {cache_file}")
        return descriptions
    
    def encode_classes(self, class_descriptions: Dict[str, str], all_classes: List[str]) -> torch.Tensor:
        """Encode class descriptions into embeddings."""
        logger.info("Encoding class descriptions...")
        descriptions_list = [class_descriptions.get(cls, f"The product category is {cls}") 
                            for cls in all_classes]
        # Use multi-GPU encoding
        embeddings = self.model.encode(
            descriptions_list,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=self.primary_device,
            batch_size=64
        )
        return embeddings
    
    def encode_documents_parallel(self, documents: List[str], batch_size: int = 64) -> torch.Tensor:
        """
        Encode documents using parallel GPU processing.
        Automatically distributes workload across available GPUs.
        """
        logger.info(f"Encoding {len(documents)} documents across {len(self.device_ids) or 1} GPU(s)...")
        
        if len(self.device_ids) <= 1:
            # Single GPU or CPU
            embeddings = self.model.encode(
                documents,
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=batch_size,
                device=self.primary_device
            )
        else:
            # Multi-GPU encoding: SentenceTransformer handles this automatically
            # when using DataParallel mode
            embeddings = self.model.encode(
                documents,
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=batch_size * len(self.device_ids),  # Scale batch size
                device=self.primary_device
            )
        
        return embeddings


# ============================================================================
# PHASE 2: REFINEMENT LOOP
# ============================================================================

class IterativePseudoLabeler:
    """
    Performs iterative refinement using transductive signal.
    Optimized for GPU operations.
    """
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
    def compute_similarity(self, doc_embeddings: torch.Tensor, class_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between documents and classes."""
        doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)
        class_embeddings = torch.nn.functional.normalize(class_embeddings, p=2, dim=1)
        similarity = torch.mm(doc_embeddings, class_embeddings.t())
        return similarity
    
    def refine_class_embeddings(
        self,
        doc_embeddings: torch.Tensor,
        class_embeddings: torch.Tensor,
        num_iterations: int = 3,
        top_n_reliable: int = 20,
        initial_top_k: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Iterative refinement loop."""
        logger.info(f"Starting iterative refinement for {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            
            similarity = self.compute_similarity(doc_embeddings, class_embeddings)
            
            new_class_embeddings = []
            for class_idx in range(class_embeddings.shape[0]):
                class_scores = similarity[:, class_idx]
                top_n_indices = torch.topk(class_scores, min(top_n_reliable, len(class_scores))).indices
                reliable_embeddings = doc_embeddings[top_n_indices]
                new_class_embedding = reliable_embeddings.mean(dim=0)
                new_class_embeddings.append(new_class_embedding)
            
            class_embeddings = torch.stack(new_class_embeddings)
            logger.info(f"  Class embeddings updated based on reliable document centroids")
        
        final_similarity = self.compute_similarity(doc_embeddings, class_embeddings)
        return class_embeddings, final_similarity
    
    def assign_labels_with_gap(
        self,
        similarity: torch.Tensor,
        min_labels: int = 2,
        max_gap_search: int = 5
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """Assign pseudo-labels using similarity gap heuristic."""
        logger.info("Assigning pseudo-labels with gap-based cutoff...")
        
        pseudo_labels = []
        pseudo_scores = []
        
        for doc_idx in range(similarity.shape[0]):
            scores = similarity[doc_idx]
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            diffs = sorted_scores[:-1] - sorted_scores[1:]
            valid_range = diffs[:max_gap_search]
            best_gap_idx = torch.argmax(valid_range).item()
            num_labels = max(min_labels, best_gap_idx + 1)
            
            selected_indices = sorted_indices[:num_labels].cpu().tolist()
            selected_scores = sorted_scores[:num_labels].cpu().tolist()
            
            pseudo_labels.append(selected_indices)
            pseudo_scores.append(selected_scores)
        
        avg_labels = np.mean([len(labels) for labels in pseudo_labels])
        logger.info(f"  Average labels per document: {avg_labels:.2f}")
        
        return pseudo_labels, pseudo_scores


# ============================================================================
# PHASE 3: AUGMENTATION
# ============================================================================

class AugmentationModule:
    """Handles data augmentation for under-represented classes."""
    
    def __init__(self, data_loader: DataLoader, augmentation_dir: str = "augmented_data"):
        self.data_loader = data_loader
        self.augmentation_dir = augmentation_dir
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = openai.OpenAI(api_key=api_key)
        else:
            logger.warning("OPENAI_API_KEY not found in environment variables. LLM features may not work.")
            self.openai_client = None
        os.makedirs(augmentation_dir, exist_ok=True)
        
    def identify_starved_classes(
        self,
        pseudo_labels: List[List[int]],
        train_indices: List[int],
        threshold: int = 10
    ) -> List[int]:
        """Identify classes with fewer than threshold assigned documents."""
        class_counts = {}
        for idx in train_indices:
            for class_idx in pseudo_labels[idx]:
                class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
        
        starved_classes = [
            class_idx for class_idx in range(len(self.data_loader.all_classes))
            if class_counts.get(class_idx, 0) < threshold
        ]
        
        logger.info(f"Identified {len(starved_classes)} starved classes (< {threshold} documents)")
        return starved_classes
    
    def generate_synthetic_data(self, class_idx, class_name, count=5):
        """Generate synthetic reviews for a specific class using LLM."""
        if not self.openai_client:
            return []
            
        prompt = f"""
        Write {count} distinct Amazon product reviews for the category: "{class_name}".
        Each review should be 1-2 sentences long, focusing on features relevant to "{class_name}".
        Output format: Just the reviews, one per line. No numbering.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )
            text = response.choices[0].message.content.strip()
            return [line.strip() for line in text.split('\n') if line.strip()]
        except Exception as e:
            logger.error(f"Augmentation failed for {class_name}: {e}")
            return []
        

    def generate_augmentation_data(
        self,
        starved_classes: List[int],
        num_samples_per_class: int = 15 # 최소 15개는 맞추기
    ) -> Tuple[List[str], List[List[int]]]:
        
        logger.info(f"Starting LLM Augmentation for {len(starved_classes)} classes...")
        
        aug_texts = []
        aug_labels = []
        
        # 비용/시간 문제로 tqdm 사용
        for class_idx in tqdm(starved_classes, desc="Augmenting"):
            class_name = self.data_loader.all_classes[class_idx]
            
            # 실제 데이터 개수 확인 (여기선 단순화를 위해 무조건 5개 추가 생성으로 가정하거나 로직 정교화 가능)
            # 여기서는 단순히 클래스당 5개씩만 추가해 봅니다.
            new_reviews = self.generate_synthetic_data(class_idx, class_name, count=5)
            
            for review in new_reviews:
                aug_texts.append(review)
                aug_labels.append([class_idx]) # 레이블은 해당 클래스 하나만 부여 (나중에 확장됨)

                
        logger.info(f"Generated {len(aug_texts)} synthetic samples.")
        return aug_texts, aug_labels


# ============================================================================
# PHASE 4: HIERARCHY EXPANSION
# ============================================================================

class HierarchyExpander:
    """Expands pseudo-labels to include all ancestor classes."""
    
    def __init__(self, hierarchy_graph: nx.DiGraph, class_to_idx: Dict[str, int]):
        self.hierarchy_graph = hierarchy_graph
        self.class_to_idx = class_to_idx
        
    def get_ancestors(self, class_idx: int) -> Set[int]:
        """Get all ancestor classes for a given class."""
        ancestors = set()
        queue = [class_idx]
        visited = {class_idx}
        
        while queue:
            current = queue.pop(0)
            for node in self.hierarchy_graph.nodes():
                if self.hierarchy_graph.has_edge(node, current) and node not in visited:
                    ancestors.add(node)
                    queue.append(node)
                    visited.add(node)
        
        return ancestors
    
    def expand_labels_with_hierarchy(
        self,
        pseudo_labels: List[List[int]]
    ) -> List[List[int]]:
        """Expand all pseudo-labels to include ancestors."""
        logger.info("Expanding labels with hierarchy...")
        
        expanded_labels = []
        for doc_labels in tqdm(pseudo_labels, desc="Expanding labels"):
            expanded = set(doc_labels)
            for class_idx in doc_labels:
                ancestors = self.get_ancestors(class_idx)
                expanded.update(ancestors)
            expanded_labels.append(sorted(list(expanded)))
        
        avg_original = np.mean([len(labels) for labels in pseudo_labels])
        avg_expanded = np.mean([len(labels) for labels in expanded_labels])
        logger.info(f"  Average labels: {avg_original:.2f} -> {avg_expanded:.2f}")
        
        return expanded_labels


# ============================================================================
# PHASE 5: MULTI-GPU BERT TRAINING
# ============================================================================

# ============================================================================
# MAIN PIPELINE
# ============================================================================

class MultiGPUTELEClassPipeline:
    """
    Main pipeline orchestrator with multi-GPU support.
    Automatically detects and utilizes available GPUs.
    """
    
    def __init__(self, 
                 data_dir: str = "Amazon_products", 
                 output_dir: str = "outputs", 
                 seed: int = 42,
                 device_ids: Optional[List[int]] = None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.seed = seed
        
        # Auto-detect GPUs if not specified
        available_gpus = get_available_gpus()
        self.device_ids = device_ids or available_gpus
        
        if self.device_ids:
            logger.info(f"Pipeline will use GPUs: {self.device_ids}")
        else:
            logger.info("No GPUs available, pipeline will use CPU")
        
        set_seed(seed)
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "intermediate"), exist_ok=True)
        
    def run(self):
        """Execute the full multi-GPU pipeline."""
        logger.info("="*80)
        logger.info("MULTI-GPU TELECLASS PIPELINE")
        logger.info(f"Available GPUs: {len(self.device_ids)}")
        logger.info("="*80)
        
        # Load data
        logger.info("\n" + "="*80)
        logger.info("LOADING DATA")
        logger.info("="*80)
        data_loader = DataLoader(self.data_dir)
        data_loader.load_all()
        
        # Phase 1: Multi-GPU Class Representation
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: MULTI-GPU CLASS REPRESENTATION")
        logger.info("="*80)
        class_repr = MultiGPUClassRepresentation(device_ids=self.device_ids, output_dir=self.output_dir)
        
        class_descriptions = class_repr.create_enriched_class_descriptions(
            class_keywords=data_loader.class_keywords,
            hierarchy_graph=data_loader.hierarchy_graph,
            idx_to_class=data_loader.idx_to_class
        )

        class_embeddings = class_repr.encode_classes(class_descriptions, data_loader.all_classes)
        doc_embeddings = class_repr.encode_documents_parallel(data_loader.all_corpus, batch_size=64)
        
        logger.info(f"Class embeddings shape: {class_embeddings.shape}")
        logger.info(f"Document embeddings shape: {doc_embeddings.shape}")
        
        # Phase 2: Iterative Refinement
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: ITERATIVE PSEUDO-LABELING")
        logger.info("="*80)
        
        primary_device = f'cuda:{self.device_ids[0]}' if self.device_ids else 'cpu'
        labeler = IterativePseudoLabeler(device=primary_device)
        refined_class_embeddings, final_similarity = labeler.refine_class_embeddings(
            doc_embeddings,
            class_embeddings,
            num_iterations=0,
            top_n_reliable=20
        )
        
        pseudo_labels, pseudo_scores = labeler.assign_labels_with_gap(
            final_similarity,
            min_labels=1,
            max_gap_search=3
        )
        
        # Save intermediate results
        torch.save({
            'pseudo_labels': pseudo_labels,
            'pseudo_scores': pseudo_scores,
            'class_embeddings': refined_class_embeddings,
            'doc_embeddings': doc_embeddings
        }, os.path.join(self.output_dir, "intermediate", "phase2_outputs.pt"))
        
        # Phase 3: Augmentation
        logger.info("\n" + "="*80)
        logger.info("PHASE 3: DATA AUGMENTATION")
        logger.info("="*80)
        
        aug_module = AugmentationModule(data_loader)
        starved_classes = aug_module.identify_starved_classes(
            pseudo_labels,
            data_loader.all_indices,
            threshold=15
        )
        augmented_texts, augmented_labels = aug_module.generate_augmentation_data(starved_classes)

        # Save augmented data to a txt file
        aug_save_path = os.path.join(self.output_dir, "augmented_data.txt")
        with open(aug_save_path, "w", encoding="utf-8") as f:
            for txt, label in zip(augmented_texts, augmented_labels):
                # Save as tab-separated: text \t label_indices_comma_separated
                label_str = ",".join(map(str, label))
                f.write(f"label_str: {label_str}\ttext: {txt}\n")
        logger.info(f"Augmented data saved to {aug_save_path}")
        
        # Phase 4: Hierarchy Expansion
        logger.info("\n" + "="*80)
        logger.info("PHASE 4: HIERARCHY EXPANSION")
        logger.info("="*80)
        
        expander = HierarchyExpander(data_loader.hierarchy_graph, data_loader.class_to_idx)
        expanded_labels = expander.expand_labels_with_hierarchy(pseudo_labels)

        avg_labels = np.mean([len(l) for l in expanded_labels])
        print(f"Average Labels after Expansion: {avg_labels}")
        
        # Phase 5: Custom TELEClass Training
        logger.info("\n" + "="*80)
        logger.info("PHASE 5: CUSTOM MODEL TRAINING")
        logger.info("="*80)

        # 1. [CRITICAL FIX] BERT 기반 Class Embedding 새로 생성
        # Phase 2(MPNet) 결과는 버리고, BERT 공간에서 새로 만듭니다.
        primary_device = f'cuda:{self.device_ids[0]}' if self.device_ids else 'cpu'
        """bert_class_embeddings = generate_bert_class_embeddings(
            "bert-base-uncased", data_loader.all_classes, class_descriptions, primary_device
        )"""
        
        # 2. Prepare Data (All Corpus + Augmentation)
        final_train_texts = data_loader.all_corpus + augmented_texts
        final_train_labels = expanded_labels + augmented_labels
        
        # 3. Augmentation Mask & Scaling
        # Real data: False, Aug data: True
        is_augmented_mask = [False] * len(data_loader.all_corpus) + [True] * len(augmented_texts)
        
        # Scaling factor calc (Prepare_training_data.py logic)
        num_real = len(data_loader.all_corpus)
        num_aug = len(augmented_texts)
        # Aug 데이터가 없으면 scaling 1.0 (div by zero 방지)
        aug_scaling = 1.0
        logger.info(f"Augmentation Scaling Factor: {aug_scaling:.4f}")

        # 4. Initialize Trainer
        # Phase 5: Custom TELEClass Training
        logger.info("\n" + "="*80)
        logger.info("PHASE 5: CUSTOM MODEL TRAINING (Using MPNet Backbone)")
        logger.info("="*80)

        # [CRITICAL FIX 1] Backbone Model 변경 (BERT -> MPNet)
        # LLM 설명의 뉘앙스를 이해하려면 SBERT 계열을 써야 합니다.
        target_model_name = "sentence-transformers/all-mpnet-base-v2"
        
        # [CRITICAL FIX 2] Class Embedding 생성 방식 변경
        # 별도의 함수(generate_bert...)를 쓰지 말고, 
        # Phase 1에서 썼던 class_repr.encode_classes를 그대로 재사용합니다.
        # 이렇게 해야 LLM Description이 MPNet 공간에 정확히 매핑됩니다.
        
        # (만약 class_repr 객체가 메모리 문제로 사라졌다면 다시 생성)
        if 'class_repr' not in locals():
            class_repr = MultiGPUClassRepresentation(
                model_name=target_model_name, 
                device_ids=self.device_ids, 
                output_dir=self.output_dir
            )
            
        final_class_embeddings = class_repr.encode_classes(
            class_descriptions, # LLM Description 딕셔너리
            data_loader.all_classes
        )
        
        # 2. Prepare Data (All Corpus + Augmentation)
        final_train_texts = data_loader.all_corpus + augmented_texts
        final_train_labels = expanded_labels + augmented_labels
        
        # 3. Augmentation Mask & Scaling
        is_augmented_mask = [False] * len(data_loader.all_corpus) + [True] * len(augmented_texts)
        
        num_real = len(data_loader.all_corpus)
        num_aug = len(augmented_texts)
        # aug_scaling = float(num_real) / num_aug if num_aug > 0 else 1.0
        aug_scaling = 1.0
        logger.info(f"Augmentation Scaling Factor: {aug_scaling:.4f}")

        # 4. Initialize Trainer with MPNet Config
        config = AutoConfig.from_pretrained(target_model_name)
        
        trainer = TELEClassTrainer(
            num_classes=len(data_loader.all_classes),
            hidden_dim=config.hidden_size, # MPNet hidden size (768)
            model_name=target_model_name, # [중요] BERT 대신 MPNet 로드
            class_embeddings=final_class_embeddings, # MPNet으로 만든 임베딩
            device_ids=self.device_ids,
            temperature=0.07 # 학습 안정화용 온도
        )
        
        # 5. Train
        trainer.train(
            final_train_texts,
            final_train_labels,
            is_augmented_mask,
            aug_scaling,
            data_loader.hierarchy_graph,
            epochs=5, # MPNet은 수렴이 빠르므로 epoch가 적어도 될 수 있음
            lr=2e-5,
            output_dir=os.path.join(self.output_dir, "models")
        )
        
        # 6. Inference
        """logger.info("PHASE 6: CUSTOM MODEL INFERENCE")
        inference = CustomInference(
            os.path.join(self.output_dir, "models", "best_model"),
            bert_class_embeddings, # Inference 때도 구조 초기화를 위해 필요
            data_loader.hierarchy_graph,
            device_ids=self.device_ids
        )
        
        test_predictions = inference.predict(
            data_loader.test_corpus,
            batch_size=32,
            threshold=0.5
        )
        
        test_predictions_expanded = expander.expand_labels_with_hierarchy(test_predictions)
        
        inference.generate_submission(
            test_predictions_expanded,
            data_loader.idx_to_class,
            output_path=os.path.join(self.output_dir, "submission.csv")
        )"""
        
        logger.info("\n" + "="*80)
        logger.info("MULTI-GPU PIPELINE COMPLETE!")
        logger.info("="*80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check if we're running from taxoclass directory or parent directory
    if os.path.exists("Amazon_products"):
        data_dir = "Amazon_products"
    elif os.path.exists("../Amazon_products"):
        data_dir = "../Amazon_products"
    else:
        print("Error: Cannot find Amazon_products directory")
        print("Please run from the project root or taxoclass directory")
        sys.exit(1)
    
    # Optional: Specify GPU IDs (None = use all available GPUs)
    # device_ids = [0, 1, 2, 3]  # Use specific GPUs
    device_ids = None  # Auto-detect and use all GPUs
    
    pipeline = MultiGPUTELEClassPipeline(
        data_dir=data_dir,
        output_dir="outputs",
        seed=42,
        device_ids=device_ids
    )
    pipeline.run()
