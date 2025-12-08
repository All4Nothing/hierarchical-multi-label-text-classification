# Hierarchical Submission Generation Guide

## 📋 개요

`generate_submission_hierarchy.py`는 **엄격한 계층 경로 제약**을 따르는 submission 파일을 생성합니다.

### 핵심 특징
- ✅ 각 문서는 **단일 부모-자식 경로**만 가짐
- ✅ 형제 노드 동시 선택 불가 (no branching)
- ✅ 유효한 경로 형식:
  - `부모 → 자식` (2개)
  - `부모 → 자식 → 손자` (3개)

---

## 🎯 알고리즘

### 선택 로직

1. **가장 높은 확률의 클래스 선택**
   ```
   selected_path = [highest_prob_class]
   ```

2. **두 번째 높은 확률 클래스 평가**
   - 첫 번째 클래스의 **부모 또는 자식**이면 → 추가 ✅
   - 그렇지 않으면 → 건너뜀 ❌

3. **세 번째 이후 클래스 평가**
   - `확률 >= threshold` AND
   - 현재 경로의 **부모 또는 자식**이면 → 추가 ✅
   - 경로가 여전히 유효한지 검증 (no branching)

4. **최종 검증**
   - 최소 `min_labels` 개 보장
   - 최대 `max_labels` 개 제한
   - 유효한 단일 경로 확인

---

## 🚀 실행 방법

### 기본 실행
```bash
python generate_submission_hierarchy.py \
    --model_path saved_models/best_model.pt \
    --threshold 0.1 \
    --min_labels 2 \
    --max_labels 3 \
    --output submission_hierarchy.csv
```

### 매개변수 설명

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model_path` | auto-detect | 모델 체크포인트 경로 |
| `--test_corpus` | Config.TEST_CORPUS | 테스트 데이터 경로 |
| `--threshold` | 0.1 | 확률 임계값 |
| `--min_labels` | 2 | 최소 레이블 수 |
| `--max_labels` | 3 | 최대 레이블 수 |
| `--output` | submission.csv | 출력 파일명 |

---

## 📊 예시

### 예시 1: 정상적인 경로

**예측 확률**:
```
Class 10 (Level 0): 0.95  ← Highest
Class 64 (Level 1): 0.85  ← Child of 10
Class 338 (Level 2): 0.75 ← Child of 64
Class 23 (Level 0): 0.70  ← Different root (ignored)
```

**선택 과정**:
1. Class 10 선택 (highest)
2. Class 64 추가 (10의 자식) ✅
3. Class 338 추가 (64의 자식) ✅
4. 3개 도달, 종료

**최종 결과**: `[10, 64, 338]` ✅
- 유효한 경로: 10 → 64 → 338

---

### 예시 2: 형제 노드 제외

**예측 확률**:
```
Class 3 (Level 1): 0.95   ← Highest
Class 17 (Level 2): 0.90  ← Child of 15
Class 28 (Level 2): 0.85  ← Child of 15 (sibling of 17!)
```

**선택 과정**:
1. Class 3 선택 (highest)
2. Class 17 평가:
   - 3의 부모도 자식도 아님 → 건너뜀 ❌
3. Class 28 평가:
   - 3의 부모도 자식도 아님 → 건너뜀 ❌
4. 추가 탐색... (3의 부모/자식 찾기)

**최종 결과**: `[0, 3]` (3의 부모인 0 추가)
- 유효한 경로: 0 → 3

---

### 예시 3: 역방향 경로 (자식 → 부모)

**예측 확률**:
```
Class 338 (Level 2): 0.95  ← Highest (leaf)
Class 64 (Level 1): 0.85   ← Parent of 338
Class 10 (Level 0): 0.75   ← Parent of 64
```

**선택 과정**:
1. Class 338 선택 (highest)
2. Class 64 추가 (338의 부모) ✅
3. Class 10 추가 (64의 부모) ✅

**최종 결과**: `[10, 64, 338]` (레벨 순으로 정렬)
- 유효한 경로: 10 → 64 → 338

---

## 🔍 검증

### 유효한 경로

✅ **Valid Paths**:
```python
[0, 10]           # parent → child
[10, 64, 338]     # grandparent → parent → child
[0, 3, 17]        # root → level1 → level2
```

❌ **Invalid Paths** (Branching):
```python
[10, 64, 65]      # 64와 65는 10의 자식들 (siblings)
[0, 3, 28]        # 3과 28이 서로 부모-자식 관계가 아님
[10, 23, 64]      # 10과 23이 서로 다른 루트
```

### 코드 검증
```python
def is_valid_path(classes: List[int], hierarchy) -> bool:
    """Check if classes form a valid single path"""
    sorted_classes = sorted(classes, key=lambda c: hierarchy.get_level(c))
    
    # Check each consecutive pair is parent-child
    for i in range(len(sorted_classes) - 1):
        parent = sorted_classes[i]
        child = sorted_classes[i + 1]
        
        children = hierarchy.get_children(parent)
        if child not in children:
            return False  # Not a valid parent-child relationship
    
    return True
```

---

## 📈 출력 통계

실행 시 다음 통계가 출력됩니다:

```
📈 Path Statistics:
   Total samples: 19658
   Path lengths: min=2, max=3, avg=2.73
   Samples with classes below threshold: 234 (1.2%)
   
   Path length distribution:
      2 classes: 5234 (26.6%)
      3 classes: 14424 (73.4%)

🔍 Sample Outputs:
   Sample 0: [10, 64, 338] -> ['baby_products', 'gear', 'swings']
             Probs: [0.9856, 0.8234, 0.7123]
   Sample 1: [0, 179] -> ['grocery_gourmet_food', 'food']
             Probs: [0.9234, 0.7845]
```

---

## 🆚 기존 방식과 비교

### generate_submission.py (Pure Threshold)
```python
# Top-3 by probability + ancestors
Result: [3, 17, 28]  # May include siblings
```

### generate_submission_hierarchy.py (NEW)
```python
# Single hierarchical path only
Result: [0, 3, 17]   # Guaranteed single path
```

### 차이점

| 측면 | Pure Threshold | Hierarchical Path |
|------|---------------|-------------------|
| 경로 제약 | 느슨함 (조상만 추가) | 엄격함 (단일 경로) |
| 형제 노드 | 가능 | 불가능 |
| 확률 반영 | 완전 | 부분적 |
| 계층 일관성 | 보장 | 완전 보장 |

---

## 🔧 고급 사용

### Threshold 조정

```bash
# Lower threshold (더 많은 클래스 후보)
python generate_submission_hierarchy.py --threshold 0.05 --output sub_t005.csv

# Higher threshold (더 확신있는 예측만)
python generate_submission_hierarchy.py --threshold 0.3 --output sub_t030.csv
```

**권장**: 0.05 ~ 0.2 범위에서 실험

### 레이블 수 조정

```bash
# 2개만 (더 보수적)
python generate_submission_hierarchy.py --max_labels 2 --output sub_max2.csv

# 3개 목표 (기본)
python generate_submission_hierarchy.py --max_labels 3 --output sub_max3.csv
```

### 다양한 모델 비교

```bash
# Stage 3 model
python generate_submission_hierarchy.py \
    --model_path saved_models/best_model.pt \
    --threshold 0.1 \
    --output hier_stage3.csv

# Self-training model
python generate_submission_hierarchy.py \
    --model_path saved_models/self_train_iter_5.pt \
    --threshold 0.1 \
    --output hier_st5.csv
```

---

## 🐛 디버깅

### 경고 메시지

```
⚠️  Warning: Sample 123 produced invalid path: [3, 17, 28]
```

**의미**: 알고리즘이 유효하지 않은 경로를 생성함
**해결**: 코드에 버그가 있거나 계층 구조 데이터 오류

### 검증 스크립트

```python
# Validate submission file
import csv

def validate_submission(filepath, hierarchy):
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        invalid_count = 0
        
        for row in reader:
            labels = [int(x) for x in row['labels'].split(',')]
            if not is_valid_path(labels, hierarchy):
                invalid_count += 1
                print(f"Invalid: {row['id']} -> {labels}")
        
        print(f"Total invalid paths: {invalid_count}")

validate_submission('submission_hierarchy.csv', hierarchy)
```

---

## ❓ FAQ

### Q1: 왜 높은 확률 클래스가 제외되나요?
**A**: 단일 경로 제약 때문입니다. 높은 확률이어도 현재 경로의 부모/자식이 아니면 제외됩니다.

### Q2: Threshold를 낮추면 더 나은가요?
**A**: 아닙니다. Threshold는 경로를 확장할 때만 사용되며, 첫 번째 클래스는 항상 선택됩니다.

### Q3: 모든 문서가 3개 레이블을 갖나요?
**A**: 아닙니다. 경로가 짧으면 2개만 가질 수 있습니다 (예: root → child).

### Q4: Pure threshold 방식과 어느 것이 나은가요?
**A**: 데이터셋 요구사항에 따릅니다:
- **엄격한 계층 제약 필요** → Hierarchical Path
- **확률 최대 반영** → Pure Threshold

---

## 📚 관련 파일

- `generate_submission.py` - 기존 pure threshold 방식
- `utils/hierarchy.py` - 계층 구조 관리
- `class_hierarchy.txt` - 계층 데이터
- `Config.py` - 설정 파일

---

## ✅ 체크리스트

실행 전 확인사항:
- [ ] 모델 파일 존재 (`saved_models/best_model.pt`)
- [ ] 테스트 데이터 존재 (`test_corpus.txt`)
- [ ] 계층 파일 존재 (`class_hierarchy.txt`)
- [ ] Config 설정 확인
- [ ] Threshold 설정 (0.05 ~ 0.2 권장)

실행 후 확인사항:
- [ ] Submission 파일 생성됨
- [ ] 모든 샘플이 2~3개 레이블 보유
- [ ] 경로 유효성 검증 통과
- [ ] 통계 확인 (path length distribution)

---

## 🎉 요약

```bash
# 단일 명령어로 hierarchical submission 생성
python generate_submission_hierarchy.py \
    --model_path saved_models/best_model.pt \
    --threshold 0.1 \
    --output submission_hierarchy.csv
```

**특징**:
- ✅ 엄격한 단일 경로 (no branching)
- ✅ 부모-자식 관계 완전 보장
- ✅ 2~3개 레이블 자동 조정
- ✅ 독립 실행 가능

이제 계층 구조를 완벽하게 따르는 submission을 생성할 수 있습니다! 🚀
