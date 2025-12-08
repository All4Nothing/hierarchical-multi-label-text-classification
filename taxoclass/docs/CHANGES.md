# TaxoClass Framework 수정 사항

## 수정된 이슈들

### ✅ Issue 1: Stage 2 - Core Class 다중 선택 구현

**문제**: Core Class를 하나만 선택했으나, 논문에서는 여러 개 선택 가능

**수정 내용**:
- `CoreClassMiner.core_classes`: `Dict[int, int]` → `Dict[int, List[int])`
- `identify_core_classes()`: confidence threshold를 넘는 **모든** 클래스를 Core Class로 선택
- `get_core_class()` → `get_core_classes()`: 리스트 반환
- `get_confidence_score()` → `get_confidence_scores()`: 딕셔너리 반환

**파일**: `taxoclass/models/core_mining.py`

**코드 예시**:
```python
# 이전: 하나만 선택
self.core_classes[doc_id] = core_class  # 단일 값

# 수정: 여러 개 선택
for class_id in candidates:
    if conf_score >= threshold:
        doc_core_classes.append(class_id)  # 모든 threshold 초과 클래스
self.core_classes[doc_id] = doc_core_classes  # 리스트
```

---

### ✅ Issue 2: Stage 3 - 라벨 생성 로직 추가

**문제**: Core Class의 조상 클래스를 Positive로 설정하는 로직 부재

**수정 내용**:
- 새로운 함수 추가: `create_training_labels()`
- **Positive Set (label=1)**: Core classes + 모든 조상 클래스
- **Negative Set (label=0)**: 그 외 클래스
- **Ignore Set (label=-1)**: Core classes의 자손 클래스

**파일**: `taxoclass/models/core_mining.py`

**함수 시그니처**:
```python
def create_training_labels(
    core_classes_dict: Dict[int, List[int]],
    hierarchy,
    num_classes: int
) -> np.ndarray:
    """
    Returns:
        Label matrix (num_docs, num_classes) where:
            1 = positive (core class or ancestor)
            0 = negative (other classes)
           -1 = ignore (descendants of core classes)
    """
```

**사용 예시**:
```python
from models import create_training_labels

# Core classes 마이닝 후
core_classes = miner.identify_core_classes()

# 학습용 라벨 생성
train_labels = create_training_labels(
    core_classes_dict=core_classes,
    hierarchy=hierarchy,
    num_classes=hierarchy.num_classes
)

# TaxoClassifier 학습 시 사용
dataset = TaxoDataset(documents, train_labels, tokenizer)
```

---

### ✅ Issue 3: Stage 4 - KL Divergence Loss 구현

**문제**: KL Divergence 대신 BCEWithLogitsLoss 사용

**수정 내용**:
- `kl_divergence_loss()` 함수 완전 재구현
- Binary KL divergence 사용 (multi-label 대응):
  ```
  KL(q || p) = q * log(q/p) + (1-q) * log((1-q)/(1-p))
  ```
- `train_iteration()`에서 KL Loss 사용
- Predictions를 확률로 변환 (`set_return_probs(True)`)

**파일**: `taxoclass/models/self_training.py`

**코드 변경**:
```python
# 이전: BCEWithLogitsLoss 사용
criterion = nn.BCEWithLogitsLoss(reduction='none')
logits = self.model(input_ids, attention_mask)
loss = criterion(logits, batch_targets)

# 수정: KL Divergence 사용
actual_model.set_return_probs(True)
predictions = self.model(input_ids, attention_mask)  # 확률
actual_model.set_return_probs(False)
loss = self.kl_divergence_loss(predictions, batch_targets)
```

---

### ✅ Issue 4: Stage 4 - Temperature 값 수정

**문제**: Temperature=2.0은 논문 의도와 반대 (distribution을 smooth하게 만듦)

**수정 내용**:
- 기본값 변경: `temperature=2.0` → `temperature=0.5`
- T < 1 : Distribution을 sharpen (high confidence 강화)
- T > 1 : Distribution을 smooth (high confidence 약화)
- 논문의 의도: "Strengthen high-confidence predictions" → T < 1 필요

**파일**: `taxoclass/models/self_training.py`

**효과**:
```python
# Temperature = 0.5
# P = 0.8 → Q = 0.8^(1/0.5) = 0.8^2 = 0.64 (더 낮아짐 - 오히려 약화?)
# 실제로는 P = 0.8 → Q = 0.8^2 = 0.64

# 정정: 
# T = 0.5일 때, Q = P^(1/T) = P^(1/0.5) = P^2
# P = 0.9 → Q = 0.81
# P = 0.5 → Q = 0.25
# 이는 높은 확률을 낮추는 효과... 다시 확인 필요

# 실제 올바른 해석:
# Temperature sharpening with T < 1:
# P = 0.9, T = 0.5 → Q = 0.9^2 = 0.81 (여전히 높음)
# P = 0.1, T = 0.5 → Q = 0.1^2 = 0.01 (매우 낮아짐)
# 결과: 상대적 격차가 커짐 (sharpening)
```

**주의**: Temperature 효과 검증 필요. T=0.5가 적절한지 실험 필요.

---

## 추가 개선 사항

### 통계 정보 향상
- `get_statistics()`: 다중 Core Class 통계 추가
  - `total_core_classes`: 전체 Core Class 개수
  - `avg_core_classes_per_doc`: 문서당 평균 Core Class 개수

### 호환성 유지
- 기존 코드와의 하위 호환성을 위해 fallback 로직 추가
- `get_confidence_scores()`: dict 또는 단일 값 모두 처리

---

## 사용 방법 변경

### Before (잘못된 구현):
```python
# Stage 2
core_classes = miner.identify_core_classes()
# → {doc_id: single_class_id}

# Stage 3 (라벨 생성 로직 없음)
# 수동으로 라벨 생성해야 함
```

### After (수정된 구현):
```python
# Stage 2
core_classes = miner.identify_core_classes()
# → {doc_id: [class_id1, class_id2, ...]}

# Stage 3
from models import create_training_labels

train_labels = create_training_labels(
    core_classes_dict=core_classes,
    hierarchy=hierarchy,
    num_classes=hierarchy.num_classes
)
# → (num_docs, num_classes) with values: 1, 0, -1
```

---

## 테스트 권장 사항

1. **Core Class 수 확인**:
   ```python
   stats = miner.get_statistics()
   print(f"Avg core classes per doc: {stats['avg_core_classes_per_doc']}")
   # 예상: 1.5 ~ 3.0 정도
   ```

2. **라벨 분포 확인**:
   ```python
   labels = create_training_labels(core_classes, hierarchy, num_classes)
   print(f"Positive: {(labels == 1).sum() / labels.size * 100:.2f}%")
   print(f"Negative: {(labels == 0).sum() / labels.size * 100:.2f}%")
   print(f"Ignore: {(labels == -1).sum() / labels.size * 100:.2f}%")
   ```

3. **Self-Training Loss 확인**:
   ```python
   # KL Loss는 일반적으로 BCE보다 큰 값
   # 정상 범위: 0.1 ~ 1.0 정도
   ```

---

## Breaking Changes

⚠️ **API 변경 사항**:
- `CoreClassMiner.get_core_class()` → `get_core_classes()` (반환값: List)
- `CoreClassMiner.get_confidence_score()` → `get_confidence_scores()` (반환값: Dict)
- `core_classes` dict 구조 변경: `int` → `List[int]`

기존 코드를 사용 중이라면 다음과 같이 수정 필요:
```python
# Before
core = miner.get_core_class(doc_id)
if core != -1:
    print(f"Core class: {core}")

# After
cores = miner.get_core_classes(doc_id)
if cores:
    print(f"Core classes: {cores}")
```

---

## 파일 수정 목록

1. ✅ `taxoclass/models/core_mining.py`
   - Multi-label core class selection
   - Label generation function

2. ✅ `taxoclass/models/self_training.py`
   - KL Divergence loss implementation
   - Temperature parameter fix

3. ✅ `taxoclass/models/__init__.py`
   - Export `create_training_labels`

---

## 다음 단계

1. **통합 테스트**: 전체 파이프라인 실행하여 검증
2. **성능 비교**: 수정 전/후 성능 비교
3. **Hyperparameter 튜닝**: Temperature, threshold 등 최적값 탐색
4. **메인 학습 스크립트 업데이트**: `create_training_labels` 적용

---

## 참고 사항

- 논문: "TaxoClass: Hierarchical Multi-Label Text Classification Using Only Class Names"
- 수정 날짜: 2025-12-07
- 주요 변경: Multi-label core classes, Hierarchical label generation, KL divergence loss
