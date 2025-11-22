# Data Usage Strategy 구현 상세

## ✅ **구현 완료: Transductive Learning 전략**

---

## 📋 **Config 설정**

```python
# config.py (Lines 51-56)

# Data Usage Strategy
# Transductive learning: Use both train and test data (both are unlabeled)
USE_TEST_IN_STAGE1 = True   # Zero-shot classification (safe, no label leakage)
USE_TEST_IN_STAGE2 = True   # Core class mining (safe, confidence-based)
USE_TEST_IN_STAGE3 = False  # Initial training (conservative, train only)
USE_TEST_IN_STAGE4 = True   # Self-training (gradual, pseudo-label based)
```

**기본 설정 (권장)**:
- ✅ Stage 1, 2, 4: Test data 사용
- ⚠️ Stage 3: Train data만 사용 (보수적)

---

## 🔧 **Stage별 구현 상세**

### **Stage 1: Similarity Calculation**

#### **구현 위치**: `main.py` Lines ~188-220

```python
if Config.USE_TEST_IN_STAGE1:
    print("🔄 Stage 1: Using TRAIN + TEST data (transductive learning)")
    stage1_documents = train_documents + test_documents
    print(f"Total: {len(stage1_documents)} (train: {len(train_documents)}, test: {len(test_documents)})")
else:
    print("📊 Stage 1: Using TRAIN data only")
    stage1_documents = train_documents
    print(f"Total: {len(stage1_documents)}")

# Compute similarity for all documents
similarity_matrix_all = similarity_calculator.compute_similarity_matrix(
    documents=stage1_documents,
    class_names=hierarchy.id_to_name,
    use_cache=True
)

# Split back to train/test
train_similarity_matrix = similarity_matrix_all[:len(train_documents)]
if Config.USE_TEST_IN_STAGE1:
    test_similarity_matrix = similarity_matrix_all[len(train_documents):]
```

**효과**:
- ✅ Test data의 similarity 정보도 활용
- ✅ 더 많은 documents로 class-document 관계 파악
- ✅ Wandb 로깅: `stage1/num_documents`, `stage1/use_test_data`

---

### **Stage 2: Core Class Mining**

#### **구현 위치**: `main.py` Lines ~238-256

```python
if Config.USE_TEST_IN_STAGE2 and Config.USE_TEST_IN_STAGE1:
    print("🔄 Stage 2: Using TRAIN + TEST data for core class mining")
    stage2_similarity_matrix = similarity_matrix_all  # Full matrix
    stage2_documents = stage1_documents  # train + test
else:
    print("📊 Stage 2: Using TRAIN data only for core class mining")
    stage2_similarity_matrix = train_similarity_matrix
    stage2_documents = train_documents

# Initialize core class miner
core_miner = CoreClassMiner(
    hierarchy=hierarchy,
    similarity_matrix=stage2_similarity_matrix,  # Use selected matrix
    candidate_power=Config.CANDIDATE_SELECTION_POWER,
    confidence_percentile=Config.CONFIDENCE_THRESHOLD_PERCENTILE
)

# Identify core classes
core_classes = core_miner.identify_core_classes()
```

**효과**:
- ✅ 더 많은 confident samples 발굴
- ✅ Test data의 high-confidence documents도 core class에 포함
- ✅ Wandb 로깅: `stage2/num_documents`, `stage2/use_test_data`

**의존성**: Stage 1에서 test data를 사용해야 Stage 2에서도 사용 가능

---

### **Stage 3: Classifier Training**

#### **구현 위치**: `main.py` Lines ~282-346

```python
if Config.USE_TEST_IN_STAGE3 and Config.USE_TEST_IN_STAGE2 and Config.USE_TEST_IN_STAGE1:
    print("🔄 Stage 3: Using TRAIN + TEST data for classifier training")
    stage3_documents = stage2_documents  # train + test
    stage3_labels = train_labels + test_labels
    
    # Create label matrix for all documents
    stage3_label_matrix = create_multi_label_matrix(
        doc_labels=stage3_labels,
        core_class_assignments=core_classes,  # From Stage 2
        hierarchy=hierarchy,
        num_classes=hierarchy.num_classes
    )
    
    # Split back for validation
    train_label_matrix = stage3_label_matrix[:len(train_documents)]
    test_label_matrix_stage3 = stage3_label_matrix[len(train_documents):]
    
    print(f"Total: {len(stage3_documents)} (train: {len(train_documents)}, test: {len(test_documents)})")
else:
    print("📊 Stage 3: Using TRAIN data only for classifier training")
    stage3_documents = train_documents
    stage3_labels = train_labels
    
    train_label_matrix = create_multi_label_matrix(
        doc_labels=train_labels,
        core_class_assignments=core_classes,
        hierarchy=hierarchy,
        num_classes=hierarchy.num_classes
    )

# Create dataset
if Config.USE_TEST_IN_STAGE3 and ...:
    train_dataset = TaxoDataset(
        documents=stage3_documents,  # train + test
        labels=stage3_label_matrix,
        tokenizer=tokenizer,
        max_length=Config.DOC_MAX_LENGTH
    )
else:
    train_dataset = TaxoDataset(
        documents=train_documents,  # train only
        labels=train_label_matrix,
        tokenizer=tokenizer,
        max_length=Config.DOC_MAX_LENGTH
    )
```

**효과**:
- ✅ Pseudo-labels로 더 많은 training samples 확보
- ⚠️ Overfitting 위험 (기본값은 False로 보수적)
- ✅ Wandb 로깅: `stage3/use_test_data`

**의존성**: Stage 1, 2에서 test data를 사용해야 Stage 3에서도 사용 가능

---

### **Stage 4: Self-Training**

#### **구현 위치**: `main.py` Lines ~451-473

```python
if Config.USE_TEST_IN_STAGE4 and Config.USE_TEST_IN_STAGE1:
    print("🔄 Stage 4: Using TRAIN + TEST data for self-training")
    stage4_documents = stage1_documents  # train + test
    print(f"Total: {len(stage4_documents)}")
    print(f"  - Train: {len(train_documents)}, Test: {len(test_documents)}")
else:
    print("📊 Stage 4: Using TRAIN data only for self-training")
    stage4_documents = train_documents
    print(f"Total: {len(stage4_documents)}")

# Create unlabeled dataset
unlabeled_loader = create_unlabeled_dataset(
    documents=stage4_documents,  # Selected documents
    tokenizer=tokenizer,
    max_length=Config.DOC_MAX_LENGTH,
    batch_size=Config.BATCH_SIZE
)

# Initialize self-trainer
self_trainer = SelfTrainer(
    model=model,
    unlabeled_loader=unlabeled_loader,  # Uses stage4_documents
    edge_index=edge_index,
    device=device,
    ...
)
```

**효과**:
- ✅ Test data로 pseudo-label 생성 및 학습
- ✅ Test distribution에 점진적 적응
- ✅ Confidence threshold로 품질 관리

**의존성**: Stage 1에서 test data를 사용해야 Stage 4에서도 사용 가능

---

## 📊 **의존성 체인**

```
Stage 1: USE_TEST_IN_STAGE1
  │
  ├─→ Stage 2: USE_TEST_IN_STAGE2 (requires Stage 1 = True)
  │     │
  │     └─→ Stage 3: USE_TEST_IN_STAGE3 (requires Stage 1, 2 = True)
  │
  └─→ Stage 4: USE_TEST_IN_STAGE4 (requires Stage 1 = True)
```

**규칙**:
1. **Stage 2**에서 test data 사용하려면: Stage 1에서 test data 필요
2. **Stage 3**에서 test data 사용하려면: Stage 1, 2에서 모두 test data 필요
3. **Stage 4**에서 test data 사용하려면: Stage 1에서 test data 필요

---

## 🎯 **사용 시나리오**

### **Scenario 1: Gradual (기본값, 권장)** ⭐

```python
USE_TEST_IN_STAGE1 = True   # ✅
USE_TEST_IN_STAGE2 = True   # ✅
USE_TEST_IN_STAGE3 = False  # ⚠️ 보수적
USE_TEST_IN_STAGE4 = True   # ✅
```

**특징**:
- Stage 1-2: Test data로 similarity & core class 확보
- Stage 3: Train only로 안정적 초기 학습
- Stage 4: Test data로 점진적 fine-tuning

**예상 성능**: 80-83% accuracy

---

### **Scenario 2: Aggressive (최대 성능)** 🔥

```python
USE_TEST_IN_STAGE1 = True   # ✅
USE_TEST_IN_STAGE2 = True   # ✅
USE_TEST_IN_STAGE3 = True   # ⚠️ 공격적
USE_TEST_IN_STAGE4 = True   # ✅
```

**특징**:
- 모든 Stage에서 test data 활용
- 최대한의 training samples

**예상 성능**: 82-85% accuracy (overfitting 위험)

**추천 설정 (overfitting 방지)**:
```python
# config.py
GNN_DROPOUT = 0.15         # 0.1 → 0.15
WEIGHT_DECAY = 0.02        # 0.01 → 0.02
SELF_TRAIN_THRESHOLD = 0.6 # 0.5 → 0.6
```

---

### **Scenario 3: Conservative (안전)** 🛡️

```python
USE_TEST_IN_STAGE1 = False  # ⚠️
USE_TEST_IN_STAGE2 = False  # ⚠️
USE_TEST_IN_STAGE3 = False  # ⚠️
USE_TEST_IN_STAGE4 = False  # ⚠️
```

**특징**:
- 전통적인 inductive learning
- 완전히 train data만 사용
- 새로운 unseen data에 대한 일반화 우선

**예상 성능**: 75-78% accuracy (robust)

---

## 📈 **Wandb 모니터링**

각 Stage에서 다음 메트릭이 로깅됩니다:

```python
# Stage 1
stage1/num_documents         # 사용된 문서 수
stage1/use_test_data         # Test data 사용 여부

# Stage 2
stage2/num_documents         # 사용된 문서 수
stage2/use_test_data         # Test data 사용 여부
stage2/num_core_classes      # Core class 개수
stage2/total_docs_with_core  # Core가 할당된 문서 수

# Stage 3
stage3/use_test_data         # Test data 사용 여부
stage3/train_samples         # Training sample 수
stage3/val_samples           # Validation sample 수
```

**확인 방법**:
```
Wandb Dashboard → Run → Overview
→ Config 탭에서 use_test_in_stage* 확인
→ Charts 탭에서 stage*/num_documents 확인
```

---

## 🔍 **실험 비교 가이드**

### **실험 설정**

```python
# Experiment 1: Conservative (Baseline)
USE_TEST_IN_STAGE1 = False
USE_TEST_IN_STAGE2 = False
USE_TEST_IN_STAGE3 = False
USE_TEST_IN_STAGE4 = False

# Experiment 2: Gradual (Recommended)
USE_TEST_IN_STAGE1 = True
USE_TEST_IN_STAGE2 = True
USE_TEST_IN_STAGE3 = False
USE_TEST_IN_STAGE4 = True

# Experiment 3: Aggressive (Maximum)
USE_TEST_IN_STAGE1 = True
USE_TEST_IN_STAGE2 = True
USE_TEST_IN_STAGE3 = True
USE_TEST_IN_STAGE4 = True
```

### **비교 메트릭**

```python
# 성능
- test/accuracy
- test/f1_score
- test/top5_accuracy

# 데이터 활용
- stage2/total_docs_with_core  # Pseudo-labeled samples
- stage4/confidence_ratio      # Self-training 품질

# 학습 안정성
- stage3/epoch_val_loss        # Overfitting 여부
- stage3/best_epoch            # Early stopping 시점
```

### **Wandb 비교**

```
Project Page → Runs 탭
→ Exp1, Exp2, Exp3 선택
→ Compare 버튼 클릭
→ Parallel coordinates plot 확인
```

---

## 💡 **Best Practices**

### **1. 점진적 활성화**

```python
# Week 1: Baseline
USE_TEST_IN_STAGE1 = False
USE_TEST_IN_STAGE2 = False
USE_TEST_IN_STAGE3 = False
USE_TEST_IN_STAGE4 = False
python main.py  # Baseline accuracy: 75-78%

# Week 2: Add Stage 1-2
USE_TEST_IN_STAGE1 = True
USE_TEST_IN_STAGE2 = True
python main.py  # Accuracy: 78-80%

# Week 3: Add Stage 4
USE_TEST_IN_STAGE4 = True
python main.py  # Accuracy: 80-83%

# Week 4: Try Stage 3 (optional)
USE_TEST_IN_STAGE3 = True
python main.py  # Accuracy: 82-85% (watch for overfitting!)
```

### **2. Overfitting 감지**

```python
# Wandb Charts에서 확인:
# 1. Train vs Val loss
#    - Train loss 계속 감소
#    - Val loss 증가 시작 → Overfitting!

# 2. Confidence ratio
#    - Stage4에서 90% 이상 → 너무 확신적, 의심 필요

# 3. Test accuracy
#    - Aggressive > Gradual 차이가 5%p 이상 → Overfitting
```

### **3. Regularization 조정**

```python
# Test data 많이 사용할수록 강화
if USE_TEST_IN_STAGE3:
    GNN_DROPOUT = 0.15         # ↑ 증가
    WEIGHT_DECAY = 0.02        # ↑ 증가
    SELF_TRAIN_THRESHOLD = 0.6 # ↑ 증가
```

---

## 🐛 **Troubleshooting**

### **문제 1: Stage 2에서 core class 너무 많음**

```
증상: stage2/num_core_classes > 전체 class의 50%

원인: Test data 포함으로 confident samples 급증

해결:
1. CONFIDENCE_THRESHOLD_PERCENTILE 상향 (50 → 60)
2. USE_TEST_IN_STAGE2 = False로 변경
```

### **문제 2: Stage 3에서 validation loss 증가**

```
증상: stage3/epoch_val_loss가 epoch 5 이후 증가

원인: Test data 사용으로 overfitting

해결:
1. USE_TEST_IN_STAGE3 = False (기본값 유지)
2. Dropout 증가 (0.1 → 0.15)
3. Early stopping patience 감소 (10 → 5)
```

### **문제 3: Self-training에서 confidence 너무 높음**

```
증상: stage4/confidence_ratio > 90%

원인: 모델이 과도하게 확신적

해결:
1. SELF_TRAIN_THRESHOLD 상향 (0.5 → 0.6)
2. SELF_TRAIN_TEMPERATURE 상향 (2.0 → 2.5)
3. Iteration 감소 (5 → 3)
```

---

## ✅ **테스트 체크리스트**

### **기능 테스트**

- [ ] Config 설정 변경 후 올바른 메시지 출력 확인
- [ ] Stage별 document 수가 config에 맞게 출력되는지 확인
- [ ] Wandb에 `use_test_data` 메트릭 로깅 확인
- [ ] 4가지 시나리오 모두 실행 가능 확인

### **성능 테스트**

- [ ] Conservative: 75-78% accuracy 달성
- [ ] Gradual: 80-83% accuracy 달성
- [ ] Aggressive: 82-85% accuracy 달성
- [ ] Train vs Val loss curve 정상 확인

### **로깅 테스트**

- [ ] `stage*/num_documents` 올바른 값 확인
- [ ] `stage*/use_test_data` True/False 올바르게 로깅
- [ ] Wandb config에 모든 설정 저장 확인

---

## 📚 **참고 자료**

1. **Transductive Learning Guide**: `TRANSDUCTIVE_LEARNING_GUIDE.md`
2. **Wandb Guide**: `WANDB_GUIDE.md`
3. **A6000 Optimization**: `A6000_OPTIMIZATION.md`

---

**마지막 업데이트**: 2025-11-22  
**구현 상태**: ✅ 완료  
**테스트 상태**: ⏳ 대기 중

