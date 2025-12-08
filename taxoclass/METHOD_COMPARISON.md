# Inference Method Comparison

## 문제 분석: Sample 1 예측 불일치

### 관찰
**모델 예측 (Top-5)**:
```
Sample 1: [3, 17, 28, 34, 4]
Probs:    [0.9999647, 0.99920005, 0.9982666, 0.9858106, 0.9663014]
```

**최종 결과 (submission_t03.csv)**:
```
Sample 1: "15,17,56"
```

### 원인
현재 구현의 **Leaf-centric 후처리**가 원인:
1. Threshold 0.3으로 많은 클래스 선택 (3, 17, 28, 34, 4, 15, 56, ...)
2. 조상 추가하여 closure 생성
3. **Leaf 노드 중 최고 확률** 선택 (예: 56)
4. 56의 조상 경로만 사용: [10, 15, 17, 56]
5. 가장 깊은 3개 선택: [15, 17, 56]
6. 높은 확률의 클래스 3, 28, 34는 경로에 없어서 **제외**

---

## 두 가지 방식 비교

### Method 1: 현재 방식 (Leaf-centric + Post-processing)

#### 코드 흐름
```python
# 1. Threshold selection
predicted_classes = np.where(probs >= threshold)[0]

# 2. Add ancestors (closure)
closure = set()
for cid in predicted_classes:
    closure.add(cid)
    closure.update(hierarchy.get_ancestors(cid))

# 3. Pick best LEAF from closure
leaf_candidates = [c for c in closure if c in leaves_set]
best_leaf = max(leaf_candidates, key=lambda c: probs[c])

# 4. Get path from root to best_leaf
path_nodes = hierarchy.get_ancestors(best_leaf) + [best_leaf]
path_nodes = sorted(path_nodes, key=lambda c: hierarchy.get_level(c))

# 5. Take deepest max_labels nodes
if len(path_nodes) >= max_labels:
    selected = path_nodes[-max_labels:]  # Deepest nodes
elif len(path_nodes) == 1:
    # Add extra classes
    ...
else:
    selected = path_nodes[:]
```

#### 특징
- ✅ **Leaf 노드 보장**: 항상 구체적인 클래스 포함
- ✅ **단일 계층 경로**: 명확한 Root → Leaf 경로
- ✅ **깊이 우선**: 더 구체적인 클래스 선호
- ❌ **높은 확률 무시**: 경로 밖의 높은 확률 클래스 제외
- ❌ **Multi-label 제한**: 단일 경로만 선택
- ❌ **논문과 다름**: 명시적 path selection은 논문에 없음

#### 적합한 경우
- 구체적인 카테고리가 중요 (상품 분류 등)
- Leaf 예측이 필수
- Kaggle 등 실용적 목적

---

### Method 2: 순수 Threshold (Pure Threshold + Ancestor Closure)

#### 코드 흐름
```python
# 1. Threshold selection
predicted = np.where(probs >= threshold)[0]

# 2. Add ancestors for consistency
closure = set()
for cls_id in predicted:
    closure.add(cls_id)
    closure.update(hierarchy.get_ancestors(cls_id))

# 3. Simple top-K by probability
closure_sorted = sorted(closure, key=lambda c: probs[c], reverse=True)
selected = closure_sorted[:max_labels]
```

#### 특징
- ✅ **논문에 충실**: Multi-label binary classification 원칙
- ✅ **확률 우선**: 가장 높은 확률 클래스 선택
- ✅ **계층 일관성**: 조상 포함으로 보장
- ✅ **Multi-label 완전 활용**: 여러 경로 가능
- ⚠️ **Leaf 미보장**: 상위 레벨만 선택될 수 있음
- ⚠️ **다양한 깊이**: 서로 다른 레벨 혼합

#### 적합한 경우
- 논문 재현이 목적
- 다양한 추상화 레벨 필요
- 모델 확신을 최대한 반영

---

## 실행 예시

### Method 1: Leaf-centric (기본)
```bash
python generate_submission.py \
    --threshold 0.3 \
    --min_labels 2 \
    --max_labels 3 \
    --output submission_leafcentric.csv
```

**예상 결과 (Sample 1)**:
```
"15,17,56"  # Leaf-centric path
```

### Method 2: Pure Threshold (논문 방식)
```bash
python generate_submission.py \
    --threshold 0.3 \
    --min_labels 2 \
    --max_labels 3 \
    --pure_threshold \
    --output submission_pure.csv
```

**예상 결과 (Sample 1)**:
```
"3,17,28"  # Highest probability classes
```

---

## 구체적 비교 (Sample 1)

### 가정
**Hierarchy**:
```
Level 0: 0 (grocery), 10 (baby), 23 (health)
Level 1: 3, 4, 15, ...
Level 2: 17, 28, 34, ...
Level 3: 56 (leaf), ...
```

**Predictions (threshold=0.3)**:
```
Class 3:  0.9999 (Level 1)
Class 17: 0.9992 (Level 2) 
Class 28: 0.9982 (Level 2)
Class 34: 0.9858 (Level 2)
Class 4:  0.9663 (Level 1)
Class 56: 0.85   (Level 3, Leaf)
Class 15: 0.75   (Level 1)
```

### Method 1: Leaf-centric

**Step 1**: Threshold → [3, 17, 28, 34, 4, 56, 15, ...]

**Step 2**: Add ancestors → [0, 3, 10, 15, 17, 23, 28, 34, 4, 56, ...]

**Step 3**: Best leaf = 56 (highest prob among leaves)

**Step 4**: Path of 56 = [10, 15, 17, 56]

**Step 5**: Take deepest 3 = [15, 17, 56]

**Result**: `"15,17,56"` ✓

**Analysis**:
- ✅ Includes leaf (56)
- ✅ Clear path (10→15→17→56)
- ❌ Misses highest prob classes (3, 28, 34)
- ❌ Class 15 has lower prob (0.75) than 3 (0.9999)

---

### Method 2: Pure Threshold

**Step 1**: Threshold → [3, 17, 28, 34, 4, 56, 15, ...]

**Step 2**: Add ancestors
- 3 → add 0
- 17 → add 10, 15
- 28 → add 10, 15
- 56 → add 10, 15, 17
- etc.

**Step 3**: Closure = [0, 3, 10, 15, 17, 23, 28, 34, 4, 56, ...]

**Step 4**: Sort by prob and take top-3:
1. Class 3:  0.9999
2. Class 17: 0.9992
3. Class 28: 0.9982

**Result**: `"3,17,28"`

**Analysis**:
- ✅ Highest probability classes
- ✅ Hierarchically consistent (ancestors included if needed)
- ⚠️ No leaf node
- ✅ Reflects model confidence

---

## 성능 비교 예상

### Method 1 (Leaf-centric)
**장점**:
- 구체적 예측 (실용적)
- 일관된 경로

**단점**:
- 모델 확신 무시
- 특정 경우 suboptimal

**예상 Metrics**:
- Precision: Medium-High
- Recall: Medium
- F1: Medium-High

### Method 2 (Pure Threshold)
**장점**:
- 모델 확신 반영
- 논문에 충실

**단점**:
- Leaf 미보장
- 다양한 깊이

**예상 Metrics**:
- Precision: High (confident predictions)
- Recall: Depends on threshold
- F1: Potentially higher

---

## 권장 사항

### 실험 프로토콜
```bash
# 1. Both methods with multiple thresholds
for t in 0.3 0.4 0.5 0.6 0.7; do
    # Leaf-centric
    python generate_submission.py \
        --threshold $t \
        --output "results/leafcentric_t${t}.csv"
    
    # Pure threshold
    python generate_submission.py \
        --threshold $t \
        --pure_threshold \
        --output "results/pure_t${t}.csv"
done

# 2. Compare results
python compare_submissions.py \
    --method1 results/leafcentric_t0.5.csv \
    --method2 results/pure_t0.5.csv
```

### 선택 기준

**Use Leaf-centric if**:
- 데이터셋이 명확한 leaf 카테고리를 요구
- Kaggle 등 경쟁에서 구체적 예측 선호
- 계층 경로가 중요

**Use Pure Threshold if**:
- 논문 재현/비교가 목적
- 다양한 추상화 레벨이 필요
- 모델 확신을 최대한 활용하고 싶음

### 최종 제안
1. **둘 다 실험**: 검증 데이터로 성능 비교
2. **Threshold 튜닝**: 각 방식에 대해 최적 threshold 탐색
3. **Ensemble**: 두 방식의 결과를 결합하는 것도 고려

---

## 실행 명령어 요약

```bash
# 기본 (Leaf-centric)
python generate_submission.py --threshold 0.5 --output submission.csv

# 논문 방식 (Pure Threshold)
python generate_submission.py --threshold 0.5 --pure_threshold --output submission_pure.csv

# Hierarchical Confidence
python generate_submission.py --hier_confidence --confidence_threshold 0.5 --output submission_hier.csv

# Hierarchical Top-1 (코드에는 있지만 flag 없음, 후처리가 유사한 효과)
```

---

## 결론

**Sample 1의 불일치 원인**:
- Leaf-centric 후처리가 leaf 56을 우선시
- 높은 확률의 non-leaf 클래스들 (3, 28, 34) 제외
- 설계된 동작이지만 논문 방식과 다름

**해결**:
- `--pure_threshold` 옵션 추가
- 두 방식을 선택 가능하게 함
- 각 방식의 장단점 이해하고 상황에 맞게 선택

**추천**:
- 먼저 Pure Threshold로 논문 재현
- 실용적 목적이면 Leaf-centric도 시도
- 검증 데이터로 최적 방식 결정
