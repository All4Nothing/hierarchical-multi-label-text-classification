# TaxoClass Inference Strategy Analysis

## 논문의 Inference 방식

TaxoClass 논문에서는 inference 시 다음과 같은 방식을 사용합니다:

### 1. 모델 예측
- TaxoClassifier가 각 클래스에 대한 확률 `P(y_j=1|D_i)`를 출력
- GNN을 통해 계층 구조 정보가 이미 임베딩에 반영됨
- Binary classification per class (multi-label)

### 2. 논문에 명시된 Inference 방식
논문은 **threshold-based multi-label classification**을 사용:
- 각 클래스에 대해 독립적으로 확률 계산
- Threshold를 넘는 모든 클래스를 예측으로 선택
- 계층 구조는 **학습 시 GNN**을 통해 이미 반영됨

### 3. 계층 일관성 (Hierarchical Consistency)
논문은 명시적인 후처리보다는:
- **학습 단계**에서 조상 클래스를 positive로 설정하여 일관성 학습
- GNN을 통해 부모-자식 관계 학습
- 모델 자체가 계층적으로 일관된 예측을 하도록 유도

---

## 구현된 3가지 방식 비교

### Option 1: Threshold-based (Default)
```python
predicted_classes = np.where(probs >= threshold)[0].tolist()
```

**특징**:
- ✅ 논문의 기본 접근법과 일치
- ✅ 각 클래스를 독립적으로 평가
- ✅ Multi-label 특성 완전 활용
- ⚠️ 계층 일관성 보장 안됨 → 후처리 필요

**장점**:
- 모델의 원래 예측 존중
- 유연한 레이블 수 (threshold 조정 가능)
- 논문 원본 방식

**단점**:
- 계층 불일치 가능 (자식만 예측, 부모는 예측 안함)
- 레이블 수 제어 어려움

---

### Option 2: Hierarchical Top-1
```python
selected = select_hierarchical_top1(probs, level_nodes_cache, min_labels, max_labels)
```

**특징**:
- 각 레벨에서 가장 높은 확률의 클래스 1개씩 선택
- Level 0 → Level 1 → Level 2 순서대로
- 계층 경로 보장

**장점**:
- ✅ 계층 일관성 완벽 보장
- ✅ 레이블 수 제어 가능 (레벨 수 = 레이블 수)
- 명확한 계층 경로

**단점**:
- ❌ 논문 방식과 다름
- ❌ Multi-label 특성 제한 (각 레벨당 1개만)
- 같은 레벨의 다른 관련 클래스 무시

---

### Option 3: Hierarchical Confidence Path
```python
selected = select_hierarchical_confidence_path(
    probs, level_nodes_cache, confidence_threshold, min_labels, max_labels
)
```

**특징**:
- Root부터 시작하여 confidence가 높으면 다음 레벨로 확장
- **단일 경로** 선택 (하나의 계층 경로)
- Level-by-level expansion

**장점**:
- ✅ 계층 일관성 보장
- ✅ Confidence 기반으로 깊이 조절
- Uncertainty가 높으면 shallow prediction

**단점**:
- ❌ 논문 방식과 다름
- ❌ 단일 경로만 선택 (multi-label 제한)
- 여러 관련 클래스 동시 예측 불가

---

## 현재 구현의 후처리 (All 3 Options 공통)

선택 방식과 무관하게, 모든 옵션에 **후처리**가 적용됨:

```python
# 1. Closure: 선택된 클래스의 조상 추가
for cid in predicted_classes:
    closure.add(cid)
    closure.update(hierarchy.get_ancestors(cid))

# 2. Leaf-centric path 선택
leaf_candidates = [c for c in closure if c in leaves_set]
best_leaf = max(leaf_candidates, key=lambda c: probs[c])
path_nodes = hierarchy.get_ancestors(best_leaf) + [best_leaf]

# 3. 2~3 labels 강제
if len(path_nodes) >= max_labels:
    selected = path_nodes[-max_labels:]  # 가장 깊은 노드들
elif len(path_nodes) < min_labels:
    # 추가 클래스 padding
```

**문제점**:
- 원래 선택 방식의 의미가 희석됨
- 어떤 옵션을 선택해도 최종 결과는 유사할 수 있음
- 후처리가 너무 강력함

---

## 논문과의 일치도 분석

### 🥇 Threshold-based (가장 논문과 일치)

**이유**:
1. ✅ 논문의 명시적 방식
2. ✅ Multi-label binary classification 원칙
3. ✅ 각 클래스를 독립적으로 평가
4. ✅ GNN이 계층 정보를 학습했으므로 explicit hierarchy enforcement 불필요

**논문 근거**:
- Section 3.3 (Classifier Training): "binary cross entropy loss for multi-label classification"
- Section 3.4 (Self-Training): "predict labels for all documents using the trained model"
- 명시적으로 hierarchical path selection을 언급하지 않음

**추천 설정**:
```bash
python generate_submission.py \
    --threshold 0.5 \
    --min_labels 2 \
    --max_labels 3
```
(기본 threshold-based, 후처리로 계층 일관성 보장)

---

### 🥈 Hierarchical Top-1 (실용적 대안)

**적합한 경우**:
- Kaggle 등 경쟁에서 계층 일관성이 명시적으로 요구될 때
- 레이블 수를 정확히 제어해야 할 때
- 명확한 계층 경로가 중요할 때

**설정**:
```bash
python generate_submission.py \
    --threshold 0.5 \
    --min_labels 2 \
    --max_labels 3
```
(코드에 명시적 flag 없음 - 후처리가 유사한 효과)

---

### 🥉 Hierarchical Confidence Path (가장 제한적)

**적합한 경우**:
- 불확실성을 명시적으로 다루고 싶을 때
- Shallow prediction이 필요한 경우

**문제**:
- 단일 경로만 선택하므로 multi-label의 이점 손실

**설정**:
```bash
python generate_submission.py \
    --hier_confidence \
    --confidence_threshold 0.5 \
    --min_labels 2 \
    --max_labels 3
```

---

## 최종 추천

### 🎯 **논문에 가장 충실한 실행 명령어**

```bash
python generate_submission.py \
    --output submission.csv \
    --threshold 0.5 \
    --min_labels 2 \
    --max_labels 3
```

**이유**:
1. **Threshold-based가 기본** (no flags)
2. 논문의 multi-label binary classification 원칙
3. 후처리가 계층 일관성과 2~3 labels constraint 보장
4. 모델이 학습한 계층 정보 최대한 활용

---

### 🎯 **실험적 최적 설정** (Threshold 튜닝)

```bash
# Threshold를 조정하여 최적 성능 탐색
python generate_submission.py \
    --threshold 0.3 \
    --min_labels 2 \
    --max_labels 3 \
    --output submission_t03.csv

python generate_submission.py \
    --threshold 0.5 \
    --min_labels 2 \
    --max_labels 3 \
    --output submission_t05.csv

python generate_submission.py \
    --threshold 0.7 \
    --min_labels 2 \
    --max_labels 3 \
    --output submission_t07.csv
```

**Threshold 효과**:
- **Lower (0.3)**: More labels predicted → 후처리에서 더 많은 선택지
- **Medium (0.5)**: Balanced
- **Higher (0.7)**: Fewer, more confident predictions

---

### 🎯 **Hierarchical Confidence 실험** (대안)

```bash
python generate_submission.py \
    --hier_confidence \
    --confidence_threshold 0.3 \
    --min_labels 2 \
    --max_labels 3 \
    --output submission_hier.csv
```

**언제 사용**:
- Threshold-based 성능이 좋지 않을 때
- 명확한 계층 경로가 필요할 때

---

## 코드 개선 제안

### 현재 문제점
후처리가 너무 강력해서 선택 방식의 차이가 희석됨

### 제안 1: 후처리 분리
```python
parser.add_argument("--no_postprocess", action="store_true",
                    help="Disable hierarchical consistency post-processing")
```

### 제안 2: Threshold + Ancestor Closure (순수 논문 방식)
```python
# 1. Threshold-based selection
predicted = np.where(probs >= threshold)[0]

# 2. Add ancestors only (no leaf-centric path selection)
closure = set(predicted)
for cls in predicted:
    closure.update(hierarchy.get_ancestors(cls))

# 3. Simple top-k selection
selected = sorted(closure, key=lambda c: probs[c], reverse=True)[:max_labels]
```

---

## 요약

| 방식 | 논문 일치도 | Multi-label 활용 | 계층 일관성 | 추천도 |
|------|------------|-----------------|------------|--------|
| **Threshold-based** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (후처리) | 🥇 **1순위** |
| Hierarchical Top-1 | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | 🥈 2순위 |
| Hierarchical Confidence | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | 🥉 3순위 |

**결론**: **Threshold-based (기본값)** 사용이 논문에 가장 충실하며, 후처리가 실용적 제약(2~3 labels, 계층 일관성)을 보장합니다.
