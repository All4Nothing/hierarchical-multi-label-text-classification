# Inference Methods Guide

## 🎯 두 가지 Inference 방식

TaxoClass framework는 이제 두 가지 inference 방식을 지원합니다:

### Method 1: Leaf-centric + Post-processing (기본)
- 구체적인 leaf 노드 보장
- 단일 계층 경로 선택
- 실용적, Kaggle 등에 적합

### Method 2: Pure Threshold + Ancestor Closure (논문 방식)
- 논문에 충실한 구현
- 확률 우선 선택
- Multi-label 완전 활용

---

## 📊 Sample 1 예측 차이 분석

### 모델 예측
```python
Top-5 predictions:
  Class 3:  0.9999647  (Level 1)
  Class 17: 0.99920005 (Level 2)
  Class 28: 0.9982666  (Level 2)
  Class 34: 0.9858106  (Level 2)
  Class 4:  0.9663014  (Level 1)
```

### Method 1 결과: `"15,17,56"`
**과정**:
1. Threshold 0.3 → 많은 클래스 선택
2. Best leaf 선택 (예: 56)
3. 56의 경로: [10, 15, 17, 56]
4. 깊은 3개: [15, 17, 56]

**특징**:
- ✅ Leaf 보장 (56)
- ❌ 높은 확률 클래스 3, 28, 34 누락

### Method 2 결과: `"3,17,28"` (예상)
**과정**:
1. Threshold 0.3 → 많은 클래스 선택
2. 조상 추가
3. 확률 상위 3개: [3, 17, 28]

**특징**:
- ✅ 최고 확률 반영
- ⚠️ Leaf 미보장

---

## 🚀 실행 명령어

### 1. Leaf-centric (기본, 현재 구현)
```bash
python generate_submission.py \
    --threshold 0.5 \
    --min_labels 2 \
    --max_labels 3 \
    --output submission_leafcentric.csv
```

**언제 사용**:
- 구체적 카테고리가 중요
- Leaf 예측이 필수
- Kaggle 등 경쟁

---

### 2. Pure Threshold (논문 방식)
```bash
python generate_submission.py \
    --threshold 0.5 \
    --min_labels 2 \
    --max_labels 3 \
    --pure_threshold \
    --output submission_pure.csv
```

**언제 사용**:
- 논문 재현/비교
- 모델 확신 최대 활용
- 다양한 레벨 혼합 필요

---

### 3. Threshold 비교 실험
```bash
# Leaf-centric with different thresholds
for t in 0.3 0.4 0.5 0.6 0.7; do
    python generate_submission.py \
        --threshold $t \
        --output "results/leafcentric_t${t}.csv"
done

# Pure threshold with different thresholds
for t in 0.3 0.4 0.5 0.6 0.7; do
    python generate_submission.py \
        --threshold $t \
        --pure_threshold \
        --output "results/pure_t${t}.csv"
done
```

---

### 4. 두 방식 비교
```bash
# Generate submissions with both methods
python generate_submission.py \
    --threshold 0.5 \
    --output submission_method1.csv

python generate_submission.py \
    --threshold 0.5 \
    --pure_threshold \
    --output submission_method2.csv

# Compare results
python compare_methods.py \
    --file1 submission_method1.csv \
    --file2 submission_method2.csv
```

**출력 예시**:
```
SAMPLE-BY-SAMPLE COMPARISON
================================================================================

Sample 1:
  Method 1: [15, 17, 56]
  Method 2: [3, 17, 28]
  Common:   [17]
  Only in Method 1: [15, 56]
  Only in Method 2: [3, 28]

Sample 2:
  Method 1: [10, 64, 338]
  Method 2: [10, 64, 338]
  Common:   [10, 64, 338]

...

⚠️  5/10 samples differ (50.0%)
```

---

## 📈 성능 비교 가이드

### 1. 검증 데이터로 평가
```bash
# If you have validation labels
python evaluate_submission.py \
    --predictions submission_method1.csv \
    --ground_truth validation_labels.csv \
    --output eval_method1.txt

python evaluate_submission.py \
    --predictions submission_method2.csv \
    --ground_truth validation_labels.csv \
    --output eval_method2.txt
```

### 2. 비교 메트릭
- **Precision**: 예측한 것 중 정답 비율
- **Recall**: 정답 중 예측한 비율
- **F1 Score**: Harmonic mean
- **Hierarchical metrics**: Path-based evaluation

### 3. 예상 차이
| Metric | Leaf-centric | Pure Threshold |
|--------|--------------|----------------|
| Precision | Medium-High | High |
| Recall | Medium | Medium-High |
| Leaf Coverage | 100% | Varies |
| High-Prob Match | Low | High |

---

## 🔍 디버깅 & 분석

### 특정 샘플 분석
```bash
# Compare specific samples
python compare_methods.py \
    --file1 submission_method1.csv \
    --file2 submission_method2.csv \
    --samples 1,5,10,20,50
```

### 전체 분석
```bash
python compare_methods.py \
    --file1 submission_method1.csv \
    --file2 submission_method2.csv \
    --all
```

### 예측 확률 확인
```python
# In generate_submission.py, uncomment debug output:
for i in range(min(5, len(predictions))):
    top_5 = np.argsort(predictions[i])[-5:][::-1]
    top_5_probs = predictions[i][top_5]
    print(f"Sample {i} top-5: {top_5.tolist()}, probs: {top_5_probs}")
```

---

## ⚖️ 선택 가이드

### Use Leaf-centric if:
```
✓ 데이터셋이 명확한 leaf 카테고리 요구
✓ 구체적 예측이 중요 (e.g., 상품 분류)
✓ Kaggle 등 경쟁
✓ 계층 경로가 중요
```

### Use Pure Threshold if:
```
✓ 논문 재현이 목적
✓ 모델 확신을 최대한 활용
✓ 다양한 추상화 레벨 필요
✓ Multi-label 특성 완전 활용
```

### 추천 워크플로우
```bash
# 1. Both methods with default threshold
python generate_submission.py --threshold 0.5 --output m1_t05.csv
python generate_submission.py --threshold 0.5 --pure_threshold --output m2_t05.csv

# 2. Compare
python compare_methods.py --file1 m1_t05.csv --file2 m2_t05.csv

# 3. Tune threshold for best method
for t in 0.3 0.4 0.5 0.6 0.7; do
    python generate_submission.py \
        --threshold $t \
        --pure_threshold \  # or remove for leaf-centric
        --output "tuning/t${t}.csv"
done

# 4. Evaluate on validation set (if available)
# Choose best threshold

# 5. Final submission with best config
python generate_submission.py \
    --threshold 0.5 \
    --pure_threshold \
    --output final_submission.csv
```

---

## 📚 관련 문서

- **PREDICTION_ANALYSIS.md** - 원인 분석 상세
- **METHOD_COMPARISON.md** - 두 방식 심층 비교
- **INFERENCE_ANALYSIS.md** - 전체 inference 전략 분석
- **RUN_COMMANDS.md** - 모든 실행 명령어

---

## ❓ FAQ

### Q1: 어느 방식이 더 나은가요?
**A**: 데이터셋과 목적에 따라 다릅니다. 검증 데이터로 두 방식 모두 실험해보세요.

### Q2: Sample 1이 왜 [15,17,56]이 되었나요?
**A**: Leaf-centric 방식이 leaf 56을 우선 선택하고 그 경로만 사용했기 때문입니다. 상세한 분석은 `PREDICTION_ANALYSIS.md` 참조.

### Q3: Pure threshold가 논문에 더 충실한가요?
**A**: 네. 논문은 threshold-based multi-label classification을 명시하며, explicit path selection은 언급하지 않습니다.

### Q4: 두 방식을 결합할 수 있나요?
**A**: 네. Ensemble 방식으로 두 결과를 결합하거나, voting을 사용할 수 있습니다.

### Q5: Threshold를 어떻게 선택하나요?
**A**: 0.3~0.7 범위에서 실험하여 검증 성능이 가장 좋은 값을 선택하세요.

---

## 🎯 빠른 시작

```bash
# 1. 두 방식 비교
python generate_submission.py --threshold 0.5 --output method1.csv
python generate_submission.py --threshold 0.5 --pure_threshold --output method2.csv

# 2. 차이 확인
python compare_methods.py --file1 method1.csv --file2 method2.csv

# 3. 최적 방식 선택
# (검증 데이터로 평가 후)

# 4. 최종 제출
python generate_submission.py \
    --threshold 0.5 \
    --pure_threshold \
    --output final_submission.csv
```

이제 두 가지 방식을 자유롭게 실험하고 비교할 수 있습니다! 🎉
