# Prediction Analysis: Why Top Predictions Change

## 관찰된 현상

### Sample 1의 예측 과정

**모델의 원래 예측 (Top-5)**:
```
Sample 1 top-5 classes: [3, 17, 28, 34, 4]
Probabilities: [0.9999647, 0.99920005, 0.9982666, 0.9858106, 0.9663014]
```

**최종 제출 파일 (submission_t03.csv)**:
```
Sample 1: "15,17,56"
```

**문제**: 가장 높은 확률의 클래스들 (3, 28, 34, 4)이 사라지고, 중간 확률의 클래스들 (15, 56)이 나타남!

---

## 원인 분석

### 현재 후처리 과정 (generate_submission.py 451-507)

```python
# Step 1: Threshold-based 선택
predicted_classes = np.where(probs >= threshold)[0].tolist()
# 예: [3, 17, 28, 34, 4, 15, 56, ...] (많은 클래스)

# Step 2: Closure (조상 추가)
closure = set()
for cid in predicted_classes:
    closure.add(cid)
    closure.update(hierarchy.get_ancestors(cid))
# 예: [3, 17, 28, 34, 4, 15, 56, ..., 0, 10, 23, ...] (더 많아짐)

# Step 3: Leaf-centric path 선택
leaf_candidates = [c for c in closure if c in leaves_set]
best_leaf = max(leaf_candidates, key=lambda c: probs[c])
# 예: best_leaf = 56 (leaf 중 가장 높은 확률)
# 문제: 56의 확률이 3보다 낮아도, 3이 leaf가 아니면 56이 선택될 수 있음!

path_nodes = hierarchy.get_ancestors(best_leaf) + [best_leaf]
# 예: path_nodes = [0, 15, 56] (56의 조상 경로)

# Step 4: 2~3 labels 강제
if len(path_nodes) >= max_labels:
    selected = path_nodes[-max_labels:]  # [15, 56] (깊은 노드들만)
elif len(path_nodes) == 1:
    # 추가 클래스 padding
else:
    selected = path_nodes[:]  # [0, 15, 56]

# 최종: [15, 56] 또는 [0, 15, 56]
```

### 문제의 핵심

1. **Leaf-centric 전략**:
   - Best leaf를 먼저 선택하고, 그 경로만 사용
   - **높은 확률의 non-leaf 클래스들이 무시됨**

2. **Path-based 제약**:
   - 단일 경로만 선택 (best leaf의 조상)
   - Multi-label의 이점 상실

3. **깊이 우선**:
   - `path_nodes[-max_labels:]`는 가장 깊은 노드들만 선택
   - 상위 레벨의 중요한 클래스 무시

### 구체적 예시

**가정**:
- Class 3: Level 1 (높은 확률 0.9999)
- Class 17: Level 2 (높은 확률 0.9992)
- Class 56: Level 3, Leaf (중간 확률 0.85)

**문제**:
1. Closure에 [3, 17, 56, ...] 모두 포함됨
2. Leaf 중 best = 56 선택
3. Path = [Root, 15, 56] (56의 조상)
4. Class 3, 17은 56의 경로가 아니므로 **제외됨**!

---

## 현재 방식 vs 논문 방식 비교

### 현재 방식: Leaf-centric + Single Path

```python
# 1. Threshold selection
predicted = threshold_based_selection(probs, threshold)

# 2. Add ancestors
closure = add_ancestors(predicted)

# 3. Pick best LEAF and its path
best_leaf = max(leaf_nodes_in_closure, key=probs)
path = ancestors(best_leaf) + [best_leaf]

# 4. Take deepest nodes from path
selected = path[-max_labels:]
```

**장점**:
- ✅ 단일 계층 경로 보장
- ✅ Leaf 노드 보장 (구체적 예측)

**단점**:
- ❌ 높은 확률의 non-leaf 클래스 무시
- ❌ Multi-label 특성 상실 (단일 경로)
- ❌ 논문 방식과 다름

---

### 논문 방식: Threshold + Ancestor Closure

```python
# 1. Threshold selection
predicted = threshold_based_selection(probs, threshold)

# 2. Add ancestors for consistency
closure = set()
for cls in predicted:
    closure.add(cls)
    closure.update(ancestors(cls))

# 3. Simple top-k selection from closure
selected = sorted(closure, key=lambda c: probs[c], reverse=True)[:max_labels]
```

**장점**:
- ✅ 논문에 충실 (multi-label binary classification)
- ✅ 높은 확률 클래스 우선
- ✅ 계층 일관성 보장 (조상 포함)
- ✅ 여러 경로 가능 (multi-label)

**단점**:
- ⚠️ Leaf 보장 없음 (상위 레벨만 선택될 수 있음)

---

## 예측 결과 비교 (Sample 1)

### 가정
- Top predictions: [3, 17, 28, 34, 4, 15, 56, ...]
- Hierarchy:
  - Class 3: Level 1, Parent: 0
  - Class 17: Level 2, Parent: 15, Grandparent: 10
  - Class 56: Level 3 (Leaf), Parent: 17, ...

### 현재 방식 (Leaf-centric)

1. **Closure**: [3, 17, 28, 34, 4, 15, 56, 0, 10, ...]
2. **Leaf candidates**: [28, 34, 56, ...] (leaf만)
3. **Best leaf**: 56 (leaf 중 최고 확률, 하지만 전체에서는 낮음)
4. **Path**: [10, 15, 17, 56] (56의 조상 경로)
5. **Selected (max=3)**: [15, 17, 56] (가장 깊은 3개)

**결과**: `"15,17,56"` ✓ (파일과 일치)

### 논문 방식 (Pure Threshold + Top-K)

1. **Threshold (0.3)**: [3, 17, 28, 34, 4, 15, 56, ...]
2. **Add ancestors**:
   - 3 → add 0
   - 17 → add 15, 10
   - ...
3. **Closure**: [0, 3, 10, 15, 17, 28, 34, 4, 56, ...]
4. **Top-3 by probability**: [3, 17, 28] (가장 높은 확률 3개)

**결과**: `"3,17,28"` (다름!)

---

## 어느 방식이 더 나은가?

### Leaf-centric의 장점
- 구체적인 예측 (leaf node)
- 명확한 계층 경로
- 실용적 (Kaggle 등에서 선호)

### Pure Threshold의 장점
- 논문에 충실
- 모델의 확신 반영
- Multi-label 완전 활용

### 권장사항

**데이터셋 특성에 따라**:
1. **Leaf 예측이 중요한 경우** (예: 상품 분류):
   - Leaf-centric 사용
   - 구체적 카테고리가 중요

2. **다양한 레벨 예측이 필요한 경우**:
   - Pure threshold 사용
   - 추상/구체 클래스 혼합

3. **논문 재현이 목적**:
   - Pure threshold 사용

---

## 구현 계획

두 가지 옵션 제공:

### Option 1: Current (Leaf-centric) - 기본값
```bash
python generate_submission.py --threshold 0.3
```

### Option 2: Pure (논문 방식)
```bash
python generate_submission.py --threshold 0.3 --pure_threshold
```

### 코드 구조
```python
parser.add_argument("--pure_threshold", action="store_true",
                    help="Use pure threshold + ancestor closure (paper method)")

if args.pure_threshold:
    # Pure threshold + top-k
    selected = pure_threshold_method(probs, threshold, hierarchy, max_labels)
else:
    # Current leaf-centric method
    selected = leaf_centric_method(probs, threshold, hierarchy, max_labels)
```

---

## 결론

**Sample 1이 [3, 17, 28]이 아닌 [15, 17, 56]이 된 이유**:
1. Leaf-centric 전략이 leaf node 56을 우선 선택
2. 56의 조상 경로 [10, 15, 17, 56]만 고려
3. 가장 깊은 3개 [15, 17, 56] 선택
4. 높은 확률의 클래스 3, 28, 34는 경로에 없어서 **제외됨**

이는 **설계된 동작**이지만, 논문의 순수한 threshold 방식과는 다릅니다.
