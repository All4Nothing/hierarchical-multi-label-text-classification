# Transductive Learning Strategy Guide

## 🎯 Train + Test Data 활용 전략

이 프로젝트에서는 **Train data와 Test data 모두 라벨이 없는 상황**에서, 두 데이터를 함께 활용하여 모델을 학습합니다.

---

## 📚 배경: Inductive vs Transductive Learning

### **Inductive Learning (전통적 방법)**

```
┌─────────┐       ┌─────────┐       ┌─────────┐
│  Train  │──────→│  Model  │──────→│   Test  │
│ (seen)  │ Learn │ (learn) │ Pred. │ (unseen)│
└─────────┘       └─────────┘       └─────────┘

특징:
- Test set은 학습 중 전혀 보지 않음
- 일반화(Generalization)에 집중
- 새로운 데이터에 대한 예측 능력 중요
```

### **Transductive Learning (이 프로젝트)**

```
┌──────────────────┐       ┌─────────┐       ┌─────────┐
│  Train + Test    │──────→│  Model  │──────→│   Test  │
│ (all seen, no Y) │ Learn │ (learn) │ Pred. │  (seen) │
└──────────────────┘       └─────────┘       └─────────┘

특징:
- Test set도 학습 중 활용 (단, 라벨 없이)
- Test set 분포에 대한 적응(Adaptation)
- 더 많은 unlabeled data 활용
```

---

## ✅ **이 전략이 적합한 이유**

### **1. 문제 설정이 Transductive**

```python
# 주어진 조건:
# - Train corpus: 라벨 없음 (unlabeled)
# - Test corpus: 라벨 없음 (unlabeled)
# - 목표: Test corpus의 문서들을 클래스에 할당

# → Test set이 고정되어 있고, 미리 알 수 있음
# → Transductive setting에 완벽히 부합
```

### **2. Semi-Supervised Learning에 유리**

```python
# Stage 1-2: Zero-shot + Core Class Mining
# → Pseudo-labels 생성 (자동으로 라벨 추정)

# Train만 사용:
#   - Pseudo-labeled samples: ~10,000
#   - Model robustness: Medium

# Train + Test 사용:
#   - Pseudo-labeled samples: ~30,000
#   - Model robustness: High
#   - Test distribution 학습: ✅
```

### **3. 계층 구조 학습에 효과적**

```python
# Hierarchical classification:
# - Level 0 (Root) → Level 1 → ... → Level N (Leaf)

# GNN (Graph Neural Network) 사용:
# - 더 많은 문서 = 더 풍부한 class-document 관계
# - Test data 포함 시 계층 구조 학습 강화

# 예시:
# Train: "laptop" → Electronics > Computers
# Test: "gaming laptop" → 같은 경로 강화
#   → GNN이 더 확신 있게 학습
```

---

## ⚠️ **주의사항 & 리스크**

### **1. Data Leakage 방지**

```python
❌ 절대 하지 말아야 할 것:

# 1. Test label 사용 (당연히 안됨)
if test_labels:  # ❌
    model.fit(test_data, test_labels)

# 2. Test statistics를 hyperparameter tuning에 사용
best_threshold = optimize_on_test_accuracy()  # ❌

# 3. Test-specific feature engineering
if doc in test_set:  # ❌
    features = special_transform(doc)

✅ 올바른 사용:

# 1. Test data를 unlabeled data로 취급
unlabeled_data = train_data + test_data
model.fit_semi_supervised(unlabeled_data)

# 2. 모든 데이터를 동일하게 처리
for doc in all_data:
    features = transform(doc)  # 동일한 처리

# 3. Confidence-based pseudo-labeling
pseudo_labels = model.predict_with_confidence(unlabeled_data)
confident_samples = filter_by_threshold(pseudo_labels)
```

### **2. Overfitting to Test Distribution**

```python
# 리스크:
# Test set의 특이한 분포에 과적합될 수 있음

# 예시:
# Train: "electronics", "books", "clothing" (균등 분포)
# Test: "electronics" (90%), "books" (10%)
#   → 모델이 "electronics"에 과도하게 편향될 수 있음

# 해결책:
# 1. Regularization 강화
DROPOUT = 0.15        # ↑ 증가
WEIGHT_DECAY = 0.02   # ↑ 증가

# 2. Confidence threshold 보수적 설정
SELF_TRAIN_THRESHOLD = 0.6  # 높게 설정

# 3. Class balance 고려
use_class_weights = True
```

### **3. Generalization 한계**

```python
# Train + Test로 학습한 모델:
# ✅ 이 Test set에 최적화
# ⚠️ 새로운 unseen data에는 일반화 제한

# 시나리오:
# 1. 현재 Test set 예측: ✅ 최고 성능
# 2. 다음 달 새 데이터: ⚠️ 성능 저하 가능
# 3. 다른 도메인 데이터: ⚠️ 성능 저하 가능

# 대응:
# - 새 데이터 추가 시: Re-training 또는 Fine-tuning
# - 모델 재사용 시: Train-only 버전도 보관
```

---

## 🎯 **단계별 사용 전략 (권장)** ⭐

### **현재 Config 설정**

```python
# config.py에 추가된 옵션:
USE_TEST_IN_STAGE1 = True   # ✅ Zero-shot (안전)
USE_TEST_IN_STAGE2 = True   # ✅ Mining (안전)
USE_TEST_IN_STAGE3 = False  # ⚠️ Training (보수적)
USE_TEST_IN_STAGE4 = True   # ✅ Self-training (점진적)
```

### **Stage별 상세 전략**

#### **Stage 1: Zero-shot Classification** ✅

```python
USE_TEST_IN_STAGE1 = True  # 권장: True

이유:
- Zero-shot은 라벨을 전혀 사용하지 않음
- NLI 모델로 similarity만 계산
- Test data 사용해도 leakage 없음
- 더 많은 문서로 class-document similarity 파악

효과:
- Test set 분포 파악
- 모든 class에 대한 전반적 이해
```

#### **Stage 2: Core Class Mining** ✅

```python
USE_TEST_IN_STAGE2 = True  # 권장: True

이유:
- Confidence 기반 샘플 선택
- 높은 확신도를 가진 샘플만 사용
- Pseudo-label 품질이 높음
- Test data 포함 시 더 많은 training samples

효과:
- Training samples 증가 (예: 10k → 30k)
- 다양한 class에 대한 examples
- Class imbalance 완화
```

#### **Stage 3: Initial Classifier Training** ⚠️

```python
USE_TEST_IN_STAGE3 = False  # 권장: False (보수적)

이유:
- 초기 모델은 보수적으로 학습
- Train data만으로 견고한 baseline 구축
- Overfitting 방지
- Validation 성능으로 hyperparameter tuning

대안 (공격적):
- USE_TEST_IN_STAGE3 = True
- 단, Regularization 강화 필요
- Dropout: 0.15, Weight Decay: 0.02
```

#### **Stage 4: Self-Training** ✅

```python
USE_TEST_IN_STAGE4 = True  # 권장: True

이유:
- Pseudo-label로 점진적 학습
- Confidence threshold로 품질 관리
- Test distribution에 적응
- Unlabeled data의 최대 활용

효과:
- Test set에 대한 성능 최대화
- Confident samples부터 점진적 확장
- Model confidence 향상
```

---

## 📊 **실험 비교 프로토콜**

### **실험 설정**

```python
# Experiment 1: Train-only (Inductive)
USE_TEST_IN_STAGE1 = False
USE_TEST_IN_STAGE2 = False
USE_TEST_IN_STAGE3 = False
USE_TEST_IN_STAGE4 = False

# Experiment 2: Gradual (Recommended)
USE_TEST_IN_STAGE1 = True
USE_TEST_IN_STAGE2 = True
USE_TEST_IN_STAGE3 = False
USE_TEST_IN_STAGE4 = True

# Experiment 3: Aggressive (Maximum performance)
USE_TEST_IN_STAGE1 = True
USE_TEST_IN_STAGE2 = True
USE_TEST_IN_STAGE3 = True
USE_TEST_IN_STAGE4 = True
```

### **평가 지표**

```python
# 1. Test set 성능 (주 목표)
- Accuracy@1, @3, @5, @10
- Macro/Micro F1-score
- Per-level accuracy

# 2. 학습 안정성
- Training loss curve
- Validation loss (if using train/val split)
- Pseudo-label quality over iterations

# 3. 효율성
- Total training time
- Number of pseudo-labeled samples
- Convergence speed
```

### **예상 결과**

```python
# Test Accuracy (예상)
Train-only:     75-78%
Gradual:        80-83%  ⭐ (권장)
Aggressive:     82-85%  (overfitting risk)

# Generalization (unseen data)
Train-only:     Good     (70-75%)
Gradual:        Fair     (68-73%)
Aggressive:     Poor     (65-70%)

# Training Time
Train-only:     Baseline
Gradual:        +10-20%  (더 많은 데이터)
Aggressive:     +20-30%
```

---

## 💡 **실전 팁**

### **1. 점진적 도입 (Safest)**

```python
# Step 1: Baseline (Train-only)
python main.py --mode train  # with all False

# Step 2: Add Stage 1-2
USE_TEST_IN_STAGE1 = True
USE_TEST_IN_STAGE2 = True
python main.py --mode train

# Step 3: Add Stage 4
USE_TEST_IN_STAGE4 = True
python main.py --mode train

# Step 4: Compare results
# → 성능 향상 확인 후 최종 결정
```

### **2. Regularization 튜닝**

```python
# Test data 사용 시 Regularization 강화:

# Dropout 증가
GNN_DROPOUT = 0.15  # 0.1 → 0.15

# Weight decay 증가
WEIGHT_DECAY = 0.02  # 0.01 → 0.02

# Confidence threshold 상향
SELF_TRAIN_THRESHOLD = 0.6  # 0.5 → 0.6

# Temperature 조정
SELF_TRAIN_TEMPERATURE = 2.5  # 2.0 → 2.5 (smoother)
```

### **3. Pseudo-label 품질 모니터링**

```python
# Self-training 중 로깅:
def log_pseudo_label_quality(pseudo_labels, confidence_scores):
    # 1. Confidence distribution
    print(f"Mean confidence: {confidence_scores.mean():.3f}")
    print(f"Std confidence: {confidence_scores.std():.3f}")
    
    # 2. Class distribution
    class_counts = Counter(pseudo_labels)
    print(f"Class distribution: {class_counts}")
    
    # 3. High confidence ratio
    high_conf = (confidence_scores > 0.8).mean()
    print(f"High confidence ratio: {high_conf:.3f}")

# Warning signs:
# - Mean confidence < 0.5: 모델이 불확실
# - Std confidence > 0.3: 일관성 부족
# - Class imbalance > 10:1: 편향 위험
```

### **4. Early Stopping with Validation Split**

```python
# Train data를 train/val로 분리 (optional)
from sklearn.model_selection import train_test_split

train_docs, val_docs = train_test_split(
    train_data, 
    test_size=0.1,  # 10% validation
    random_state=42
)

# Val set으로 early stopping
best_val_loss = float('inf')
patience = 3
patience_counter = 0

for epoch in range(num_epochs):
    train_loss = train_one_epoch(train_docs)
    val_loss = evaluate(val_docs)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_model()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break

# 최종: train + val + test로 재학습 (optional)
final_train_data = train_docs + val_docs + test_docs
final_model = retrain(final_train_data)
```

---

## 🔬 **이론적 배경**

### **Transductive Learning의 수학적 정의**

```
Inductive Learning:
  Given: D_train = {(x_i, y_i)}_{i=1}^{n}
  Learn: f: X → Y
  Goal: Minimize E_{(x,y)~P}[L(f(x), y)]
  
Transductive Learning:
  Given: D_train = {(x_i, y_i)}_{i=1}^{n}, X_test = {x_j}_{j=1}^{m}
  Learn: f: X → Y (with knowledge of X_test)
  Goal: Minimize Σ_{j=1}^{m} L(f(x_j), y_j)
  
차이점:
- Inductive: 미래의 모든 x에 대해 일반화
- Transductive: 주어진 X_test에 대해 최적화
```

### **Semi-Supervised Learning과의 관계**

```python
# Semi-supervised learning:
# - Labeled data: small
# - Unlabeled data: large
# - Goal: Use unlabeled data to improve model

# 이 프로젝트:
# - Labeled data: 0 (pseudo-labels로 생성)
# - Unlabeled data: train + test (both large)
# - Goal: Generate pseudo-labels and learn

# 유사한 기법:
# - Pseudo-labeling
# - Self-training
# - Co-training
# - Consistency regularization
```

---

## 📈 **성능 최적화 체크리스트**

### **Before Training**
- [ ] Data distribution 분석 (train vs test)
- [ ] Class distribution 확인
- [ ] Regularization 설정 확인
- [ ] Validation strategy 결정

### **During Training**
- [ ] Pseudo-label 품질 모니터링
- [ ] Confidence distribution 로깅
- [ ] Loss curve 확인 (overfitting 여부)
- [ ] Class balance 모니터링

### **After Training**
- [ ] Test set 성능 평가
- [ ] Per-class 성능 분석
- [ ] Confidence analysis
- [ ] Error analysis (misclassified samples)

---

## 🎓 **결론 & 권장사항**

### **✅ 권장: Gradual Approach (단계별 사용)**

```python
# config.py 설정 (현재 적용됨):
USE_TEST_IN_STAGE1 = True   # ✅
USE_TEST_IN_STAGE2 = True   # ✅
USE_TEST_IN_STAGE3 = False  # ⚠️ (보수적)
USE_TEST_IN_STAGE4 = True   # ✅
```

**이유**:
1. ✅ 안전성: Overfitting 위험 최소화
2. ✅ 성능: Test set에 대해 높은 성능
3. ✅ 투명성: 각 단계별 기여도 파악 가능
4. ✅ 유연성: 필요시 Stage 3도 추가 가능

**예상 성능**:
- Test Accuracy: **80-83%**
- Training Time: Baseline + 10-20%
- Robustness: High
- Generalization: Fair

---

## 📚 **참고 문헌**

1. **Transductive Learning**:
   - Vapnik, V. (1998). Statistical Learning Theory
   - Joachims, T. (1999). Transductive Inference for Text Classification

2. **Semi-Supervised Learning**:
   - Zhu, X. & Goldberg, A. B. (2009). Introduction to Semi-Supervised Learning
   - Chapelle et al. (2006). Semi-Supervised Learning

3. **Self-Training**:
   - Yarowsky, D. (1995). Unsupervised Word Sense Disambiguation
   - Lee, D. H. (2013). Pseudo-Label: The Simple and Efficient Method

4. **Hierarchical Classification**:
   - Silla, C. N. & Freitas, A. A. (2011). A Survey of Hierarchical Classification
   - Kowsari et al. (2019). Text Classification Algorithms

---

**마지막 업데이트**: 2025-11-22  
**전략**: Transductive Learning with Gradual Test Data Integration  
**예상 Test Accuracy**: 80-83%

