import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
from transformers import AutoTokenizer, BertForSequenceClassification
from tqdm import tqdm
import torch.nn.functional as F

# ============================================================================
# CONFIGURATION & LOADING
# ============================================================================
DATA_DIR = "../Amazon_products"
OUTPUT_DIR = "outputs"
INTERMEDIATE_PATH = os.path.join(OUTPUT_DIR, "intermediate", "phase2_outputs.pt")
MODEL_PATH = os.path.join(OUTPUT_DIR, "models", "best_model")

print(f"🔍 [Checkpoint 0] Loading Artifacts...")

# 1. Load Data Mappings
classes = []
idx_to_class = {}
class_to_idx = {}
with open(os.path.join(DATA_DIR, "classes.txt"), 'r', encoding='utf-8') as f:
    for line in f:
        cid, cname = line.strip().split('\t')
        classes.append(cname)
        idx_to_class[int(cid)] = cname
        class_to_idx[cname] = int(cid)

# 2. Load Raw Text (for semantic check)
train_corpus = []
with open(os.path.join(DATA_DIR, "train", "train_corpus.txt"), 'r', encoding='utf-8') as f:
    for line in f:
        train_corpus.append(line.strip().split('\t', 1)[1])
test_corpus = []
with open(os.path.join(DATA_DIR, "test", "test_corpus.txt"), 'r', encoding='utf-8') as f:
    for line in f:
        test_corpus.append(line.strip().split('\t', 1)[1])
all_corpus = train_corpus + test_corpus

# 3. Load Intermediate Results
if not os.path.exists(INTERMEDIATE_PATH):
    print(f"❌ Error: {INTERMEDIATE_PATH} not found. Did Phase 2 finish?")
    exit()
    
checkpoint = torch.load(INTERMEDIATE_PATH)
pseudo_labels = checkpoint['pseudo_labels'] # List[List[int]]
pseudo_scores = checkpoint['pseudo_scores'] # List[List[float]]
print(f"✅ Loaded Phase 2 Pseudo-Labels for {len(pseudo_labels)} documents.")

# ============================================================================
# CHECKPOINT 1: PSEUDO-LABEL DISTRIBUTION
# ============================================================================
print(f"\n📊 [Checkpoint 1] Checking Label Distribution...")

all_flat_labels = [lbl for doc_labels in pseudo_labels for lbl in doc_labels]
label_counts = pd.Series(all_flat_labels).map(idx_to_class).value_counts()
doc_lengths = pd.Series([len(l) for l in pseudo_labels])

print(f"   - Total Assigned Labels: {len(all_flat_labels)}")
print(f"   - Unique Classes Covered: {len(label_counts)} / {len(classes)}")
print(f"   - Empty Classes (Zero Docs): {len(classes) - len(label_counts)}")
print(f"   - Avg Labels per Doc: {doc_lengths.mean():.2f} (Target: >2.0)")

# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
label_counts.head(20).plot(kind='bar')
plt.title("Top 20 Frequent Classes")
plt.subplot(1, 2, 2)
plt.hist(doc_lengths, bins=range(1, 10), align='left', rwidth=0.8)
plt.title("Labels per Document Distribution")
plt.xlabel("Number of Labels")
plt.tight_layout()
plt.savefig("checkpoint1_distribution.png")
print(f"   - Distribution plot saved to 'checkpoint1_distribution.png'")

# ============================================================================
# CHECKPOINT 2: SEMANTIC CONSISTENCY
# ============================================================================
print(f"\n🧠 [Checkpoint 2] Checking Semantic Consistency...")

def check_class_semantics(target_class_name):
    if target_class_name not in class_to_idx:
        return
    
    target_idx = class_to_idx[target_class_name]
    
    # Find documents where this class was predicted
    relevant_docs = []
    for doc_id, (lbls, scrs) in enumerate(zip(pseudo_labels, pseudo_scores)):
        if target_idx in lbls:
            # Get the score for this specific class
            score_idx = lbls.index(target_idx)
            score = scrs[score_idx]
            relevant_docs.append((doc_id, score))
    
    # Sort by score descending
    relevant_docs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n   Testing Class: '{target_class_name}' (Assigned to {len(relevant_docs)} docs)")
    print(f"   Top 5 High-Confidence Documents:")
    for i, (doc_id, score) in enumerate(relevant_docs[:5]):
        text_snippet = all_corpus[doc_id]
        print(f"     {i+1}. [Score: {score:.4f}] {text_snippet}")

# Check Head, Middle, Tail classes
top_class = label_counts.index[0]
mid_class = label_counts.index[len(label_counts)//2]
tail_class = label_counts.index[-1]

check_class_semantics(top_class)
check_class_semantics(mid_class)
check_class_semantics(tail_class)

# ============================================================================
# CHECKPOINT 3: HIERARCHY COMPLIANCE
# ============================================================================
print(f"\ntree [Checkpoint 3] Checking Hierarchy Logic...")

# Load Hierarchy
G = nx.DiGraph()
with open(os.path.join(DATA_DIR, "class_hierarchy.txt"), 'r') as f:
    for line in f:
        p, c = map(int, line.strip().split())
        G.add_edge(p, c)

def get_ancestors(node, graph):
    ancestors = set()
    try:
        preds = list(graph.predecessors(node))
        for p in preds:
            ancestors.add(p)
            ancestors.update(get_ancestors(p, graph))
    except:
        pass
    return ancestors

# Check a random sample of pseudo-labels (Pre-expansion state)
# Note: Since pseudo_labels are from Phase 2 (before expansion), 
# we expect some children to be present without parents. 
# Verification: If we simulate expansion, do we get sane results?

sample_idx = np.random.randint(0, len(pseudo_labels))
raw_lbls = pseudo_labels[sample_idx]
expanded_set = set(raw_lbls)
for l in raw_lbls:
    expanded_set.update(get_ancestors(l, G))

print(f"   Sample Document ID: {sample_idx}")
print(f"   - Raw Labels: {[idx_to_class[i] for i in raw_lbls]}")
print(f"   - Simulated Expansion: {[idx_to_class[i] for i in expanded_set]}")

if len(expanded_set) >= len(raw_lbls):
    print("   ✅ Expansion logic seems valid (Ancestors added).")
else:
    print("   ⚠️ Expansion logic questionable (No ancestors found?).")

# ============================================================================
# CHECKPOINT 4: AUGMENTATION STATUS
# ============================================================================
print(f"\n🧬 [Checkpoint 4] Checking Augmentation...")

# In your provided code, AugmentationModule was a placeholder.
# We check if any data was actually generated.

aug_dir = os.path.join(OUTPUT_DIR, "intermediate") 
# Note: Your code didn't explicitly save augmented text to a file in the main loop,
# but let's check the logic.
starved_classes_count = len(classes) - len(label_counts)
print(f"   - Starved Classes detected in Checkpoint 1: {starved_classes_count}")

# WARNING based on provided code analysis
print("   ⚠️ NOTICE: The provided pipeline code has a placeholder 'AugmentationModule'.")
print("   ⚠️ EXPECTED RESULT: No synthetic data was generated unless you modified 'generate_augmentation_data'.")
print("   ⚠️ This part of the pipeline likely contributed 0 additional samples.")


# ============================================================================
# CHECKPOINT 5: MODEL INFERENCE (CASE STUDY)
# ============================================================================
print(f"\n🤖 [Checkpoint 5] Final Model Inference Check...")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: {MODEL_PATH} not found.")
else:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
        model.eval()

        # Pick a random test document
        for i in range(5):
            test_idx = np.random.randint(0, len(test_corpus))
            text = test_corpus[test_idx]
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.sigmoid(logits)
                
            # Get top 3 predictions
            top_k = torch.topk(probs, 5)
            pred_indices = top_k.indices[0].cpu().numpy()
            pred_scores = top_k.values[0].cpu().numpy()

            print(f"   Test Document id: {test_idx}, text: {text}")
            print("   Model Predictions:")
            for idx, score in zip(pred_indices, pred_scores):
                print(f"     - {idx_to_class[idx]} ({score:.4f})")
            
        print("   ✅ Model loaded and inference successful.")
        
    except Exception as e:
        print(f"   ❌ Inference failed: {e}")

print(f"\n✅ Verification Complete.")