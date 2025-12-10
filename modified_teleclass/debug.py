import torch
import os
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt

# ============================================================================
# SETUP
# ============================================================================
DATA_DIR = "../Amazon_products"
MODEL_PATH = "outputs/models/best_model"
PHASE2_OUTPUT = "outputs/intermediate/phase2_outputs.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🕵️ STARTING INVESTIGATION on {device}...\n")

# ============================================================================
# INVESTIGATION 1: LABEL POISONING CHECK (가장 유력한 용의자)
# ============================================================================
print("🔍 [1/3] Checking Training Labels (Is data poisoned?)")

if os.path.exists(PHASE2_OUTPUT):
    checkpoint = torch.load(PHASE2_OUTPUT, map_location='cpu')
    pseudo_labels = checkpoint['pseudo_labels']
    
    # 빈도수 체크
    all_labels = [l for doc in pseudo_labels for l in doc]
    from collections import Counter
    counts = Counter(all_labels)
    
    print(f"   - Total Documents: {len(pseudo_labels)}")
    print("   - Top 5 Most Frequent Labels in Training Data:")
    for lbl, cnt in counts.most_common(5):
        ratio = cnt / len(pseudo_labels) * 100
        print(f"     Label ID {lbl}: appeared in {cnt} docs ({ratio:.1f}%)")
        
    if counts.most_common(1)[0][1] > len(pseudo_labels) * 0.9:
        print("   🚨 CRITICAL: Training data is POISONED. Almost all docs have the same label.")
        print("   -> Cause: Phase 4 Hierarchy Expansion logic might be adding a common root to everyone.")
    else:
        print("   ✅ Label distribution looks plausible (No single label covers >90%).")
else:
    print("   ⚠️ Phase 2 output not found. Skipping label check.")

# ============================================================================
# INVESTIGATION 2: INPUT INTEGRITY CHECK (토크나이저 고장 확인)
# ============================================================================
print("\n🔍 [2/3] Checking Input Tokenization (Is input identical?)")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # 서로 완전히 다른 두 문장 준비
    text_a = "baby milk bottle bpa free safe for toddler"
    text_b = "heavy duty hammer drill for construction concrete"
    
    enc_a = tokenizer(text_a, return_tensors='pt')['input_ids']
    enc_b = tokenizer(text_b, return_tensors='pt')['input_ids']
    
    print(f"   - Text A: '{text_a}' -> IDs: {enc_a[0][:5].tolist()}...")
    print(f"   - Text B: '{text_b}' -> IDs: {enc_b[0][:5].tolist()}...")
    
    if torch.equal(enc_a, enc_b):
        print("   🚨 CRITICAL: Tokenizer is BROKEN. Different inputs yield identical tokens.")
    else:
        print("   ✅ Tokenizer is working (Inputs are distinct).")
        
except Exception as e:
    print(f"   ❌ Tokenizer check failed: {e}")

# ============================================================================
# INVESTIGATION 3: MODEL WEIGHT & LOGIT CHECK (모델 사망 확인)
# ============================================================================
print("\n🔍 [3/3] Checking Model Weights & Logits (Is model dead?)")

try:
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    
    # Check Weights
    print("   - Inspecting classifier weights...")
    weights = model.classifier.weight.data
    w_mean, w_std = weights.mean().item(), weights.std().item()
    print(f"     Mean: {w_mean:.6f}, Std: {w_std:.6f}")
    
    if torch.isnan(weights).any():
        print("   🚨 CRITICAL: Model weights contain NaN. Gradient exploded.")
    elif w_std < 1e-6:
        print("   🚨 CRITICAL: Model weights are all identical/zero. Training failed completely.")
        
    # Check Logits for the two different texts
    with torch.no_grad():
        logit_a = model(enc_a.to(device)).logits
        logit_b = model(enc_b.to(device)).logits
        
    prob_a = torch.sigmoid(logit_a)[0]
    prob_b = torch.sigmoid(logit_b)[0]
    
    # 상위 3개 예측 비교
    top_a = torch.topk(prob_a, 3)
    top_b = torch.topk(prob_b, 3)
    
    print(f"   - Prediction A Top-3 Indices: {top_a.indices.tolist()}")
    print(f"   - Prediction B Top-3 Indices: {top_b.indices.tolist()}")
    
    if top_a.indices.tolist() == top_b.indices.tolist():
        print("   🚨 CRITICAL: Model predicts SAME class for completely different inputs.")
        print("   -> This confirms Model Collapse.")
    else:
        print("   ✅ Model discriminates between inputs.")

except Exception as e:
    print(f"   ❌ Model check failed: {e}")

print("\n🕵️ INVESTIGATION COMPLETE.")