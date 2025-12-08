# TaxoClass Generate Submission - 실행 명령어 가이드

## 🎯 논문에 가장 충실한 명령어 (추천)

### 기본 실행
```bash
python generate_submission.py \
    --output submission.csv \
    --threshold 0.5 \
    --min_labels 2 \
    --max_labels 3
```

**특징**:
- ✅ **Threshold-based multi-label classification** (논문 방식)
- ✅ 각 클래스를 독립적으로 평가
- ✅ Multi-label 특성 완전 활용
- ✅ 후처리로 계층 일관성 + 2~3 labels 보장

**동작 방식**:
1. 모델이 각 클래스에 대한 확률 예측
2. `threshold >= 0.5`인 모든 클래스 선택
3. 후처리:
   - 선택된 클래스의 조상 추가 (계층 일관성)
   - Leaf-centric path 선택
   - 2~3개 레이블 강제

---

## 🔬 Threshold 최적화 (성능 향상)

서로 다른 threshold로 여러 submission 생성 후 비교:

```bash
# Low threshold (더 많은 클래스 예측)
python generate_submission.py \
    --threshold 0.3 \
    --min_labels 2 \
    --max_labels 3 \
    --output submission_t030.csv

# Medium threshold (기본)
python generate_submission.py \
    --threshold 0.5 \
    --min_labels 2 \
    --max_labels 3 \
    --output submission_t050.csv

# High threshold (더 확신있는 예측만)
python generate_submission.py \
    --threshold 0.7 \
    --min_labels 2 \
    --max_labels 3 \
    --output submission_t070.csv
```

**Threshold 효과**:
- **0.3**: 더 많은 후보 → Recall 높음, Precision 낮음
- **0.5**: Balanced
- **0.7**: 확신있는 예측만 → Precision 높음, Recall 낮음

**추천**: 검증 데이터로 최적 threshold 찾기

---

## 🌲 Hierarchical Confidence 방식 (대안)

```bash
python generate_submission.py \
    --hier_confidence \
    --confidence_threshold 0.5 \
    --min_labels 2 \
    --max_labels 3 \
    --output submission_hier.csv
```

**특징**:
- Level-by-level 확장 (Root → Level 1 → Level 2)
- Confidence가 threshold 이상일 때만 다음 레벨 진입
- **단일 계층 경로** 선택

**언제 사용**:
- Threshold-based 성능이 기대에 미치지 못할 때
- 명확한 계층 경로가 필요할 때
- Uncertainty를 명시적으로 다루고 싶을 때

**Confidence Threshold 실험**:
```bash
# Conservative (shallow predictions)
python generate_submission.py \
    --hier_confidence \
    --confidence_threshold 0.7 \
    --output submission_hier_c07.csv

# Aggressive (deep predictions)
python generate_submission.py \
    --hier_confidence \
    --confidence_threshold 0.3 \
    --output submission_hier_c03.csv
```

---

## 🎲 모델 선택

기본적으로 자동 선택 (우선순위):
1. Self-training 모델 (`self_train_iter_{max}.pt`)
2. Best validation 모델 (`best_model.pt`)
3. Latest checkpoint (`checkpoint_epoch_{max}.pt`)

### 특정 모델 지정
```bash
python generate_submission.py \
    --model_path ./saved_models/self_train_iter_5.pt \
    --threshold 0.5 \
    --output submission_st5.csv
```

### 여러 모델 비교
```bash
# Stage 3 only (no self-training)
python generate_submission.py \
    --model_path ./saved_models/best_model.pt \
    --threshold 0.5 \
    --output submission_stage3.csv

# Self-training iteration 3
python generate_submission.py \
    --model_path ./saved_models/self_train_iter_3.pt \
    --threshold 0.5 \
    --output submission_st3.csv

# Self-training iteration 5 (final)
python generate_submission.py \
    --model_path ./saved_models/self_train_iter_5.pt \
    --threshold 0.5 \
    --output submission_st5.csv
```

---

## 📊 전체 실험 세트 (Grid Search)

성능 최적화를 위한 체계적 탐색:

```bash
#!/bin/bash
# run_experiments.sh

# Threshold-based experiments
for t in 0.3 0.4 0.5 0.6 0.7; do
    python generate_submission.py \
        --threshold $t \
        --min_labels 2 \
        --max_labels 3 \
        --output "submissions/threshold_${t}.csv"
done

# Hierarchical confidence experiments
for c in 0.3 0.4 0.5 0.6 0.7; do
    python generate_submission.py \
        --hier_confidence \
        --confidence_threshold $c \
        --min_labels 2 \
        --max_labels 3 \
        --output "submissions/hier_conf_${c}.csv"
done

# Model comparison
for model in ./saved_models/self_train_iter_*.pt; do
    iter=$(basename $model .pt | sed 's/self_train_iter_//')
    python generate_submission.py \
        --model_path $model \
        --threshold 0.5 \
        --output "submissions/model_iter${iter}.csv"
done
```

---

## 🔧 고급 옵션

### 커스텀 테스트 데이터
```bash
python generate_submission.py \
    --test_corpus /path/to/custom_test.txt \
    --threshold 0.5 \
    --output submission_custom.csv
```

### 레이블 수 조정
```bash
# 최소 1개, 최대 5개
python generate_submission.py \
    --threshold 0.5 \
    --min_labels 1 \
    --max_labels 5 \
    --output submission_1to5.csv
```

---

## 📋 명령어 옵션 정리

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model_path` | auto-detect | 모델 체크포인트 경로 |
| `--test_corpus` | Config.TEST_CORPUS | 테스트 데이터 경로 |
| `--output` | submission.csv | 출력 파일명 |
| `--threshold` | 0.5 | 확률 threshold (기본 방식) |
| `--min_labels` | 2 | 최소 레이블 수 |
| `--max_labels` | 3 | 최대 레이블 수 |
| `--hier_confidence` | False | Hierarchical confidence 방식 사용 |
| `--confidence_threshold` | 0.5 | Hier. confidence threshold |

---

## 🎯 최종 추천 워크플로우

### 1단계: 기본 실행
```bash
python generate_submission.py \
    --threshold 0.5 \
    --output submission_baseline.csv
```

### 2단계: Threshold 최적화
```bash
for t in 0.3 0.4 0.5 0.6 0.7; do
    python generate_submission.py \
        --threshold $t \
        --output submission_t${t}.csv
done
```

### 3단계: 최적 threshold로 여러 모델 비교
```bash
# Best threshold from step 2 (예: 0.5)
BEST_T=0.5

python generate_submission.py \
    --model_path ./saved_models/best_model.pt \
    --threshold $BEST_T \
    --output final_stage3.csv

python generate_submission.py \
    --model_path ./saved_models/self_train_iter_5.pt \
    --threshold $BEST_T \
    --output final_selftraining.csv
```

### 4단계: 최종 제출
```bash
# Self-training 모델 + 최적 threshold
python generate_submission.py \
    --threshold 0.5 \
    --output final_submission.csv
```

---

## 🔍 출력 확인

실행 후 다음과 같은 정보가 출력됩니다:

```
✅ Model loaded successfully!
Generating predictions...
100%|████████████████| 1000/1000 [00:30<00:00]

📊 Prediction Statistics:
   Prediction shape: (19658, 531)
   Prediction range: [0.0012, 0.9823]
   Prediction mean: 0.0342
   
Converting predictions to submission format...
100%|████████████████| 19658/19658 [00:05<00:00]

✅ Submission file saved: submission.csv
   Total samples: 19658
   Labels per sample: min=2, max=3, avg=2.73
   Total unique classes predicted: 412

SUBMISSION GENERATION COMPLETE!
```

---

## ❓ FAQ

### Q1: 어떤 방식이 가장 좋나요?
**A**: **Threshold-based (기본)**가 논문에 가장 충실하며, 일반적으로 좋은 성능을 보입니다.

### Q2: Threshold는 어떻게 설정하나요?
**A**: 0.3~0.7 범위에서 실험하여 validation 성능이 가장 좋은 값을 선택하세요.

### Q3: Self-training 모델을 꼭 사용해야 하나요?
**A**: 네, Stage 4 (self-training) 모델이 일반적으로 가장 좋은 성능을 보입니다.

### Q4: Hierarchical confidence는 언제 사용하나요?
**A**: Threshold-based가 계층 불일치 문제를 보일 때, 또는 명확한 단일 경로가 필요할 때 사용하세요.

### Q5: 레이블이 2~3개로 제한되는 이유는?
**A**: Kaggle competition 또는 실제 데이터셋의 요구사항입니다. 필요시 `--min_labels`, `--max_labels`로 조정 가능합니다.

---

## 📚 참고 문서

- `INFERENCE_ANALYSIS.md` - 상세 inference 전략 분석
- `FIXES_SUMMARY.md` - Framework 수정 사항
- `CHANGES.md` - 구현 변경 내역
