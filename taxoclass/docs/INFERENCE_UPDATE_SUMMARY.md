# Inference Method Update Summary

## 🔍 문제 발견

**관찰**: Sample 1의 예측 불일치
```
모델 Top-5 예측: [3, 17, 28, 34, 4] (확률: 0.9999, 0.9992, 0.9982, ...)
최종 제출 결과:   [15, 17, 56]
```

**질문**: 왜 가장 높은 확률의 클래스들이 사라졌는가?

---

## 💡 원인 분석

### 현재 구현 (Leaf-centric + Post-processing)

**동작 과정**:
1. Threshold로 많은 클래스 선택 (예: 3, 17, 28, 34, 4, 15, 56, ...)
2. 조상 추가하여 closure 생성
3. **Leaf 중 최고 확률** 선택 (예: 56)
4. 56의 조상 경로만 사용: [10, 15, 17, 56]
5. 가장 깊은 3개 선택: [15, 17, 56]
6. ❌ 높은 확률 클래스 3, 28, 34는 경로에 없어서 **제외됨**

**문제점**:
- Leaf-centric 전략이 경로 밖의 높은 확률 클래스 무시
- 단일 경로만 선택하여 multi-label 제한
- 논문의 순수한 threshold 방식과 다름

---

## ✅ 해결 방법

### 두 가지 방식 구현

#### 1. Leaf-centric (기존, 기본값)
```python
# 장점: Leaf 보장, 명확한 경로
# 단점: 높은 확률 무시, 논문과 다름
```

#### 2. Pure Threshold (추가, 논문 방식)
```python
def select_pure_threshold(probs, hierarchy, threshold, max_labels):
    # 1. Threshold selection
    predicted = np.where(probs >= threshold)[0]
    
    # 2. Add ancestors
    closure = set()
    for cls in predicted:
        closure.add(cls)
        closure.update(hierarchy.get_ancestors(cls))
    
    # 3. Top-K by probability
    return sorted(closure, key=lambda c: probs[c])[:max_labels]
```

---

## 🚀 사용 방법

### Leaf-centric (기본)
```bash
python generate_submission.py \
    --threshold 0.5 \
    --output submission.csv
```

### Pure Threshold (논문)
```bash
python generate_submission.py \
    --threshold 0.5 \
    --pure_threshold \
    --output submission_pure.csv
```

### 비교
```bash
python compare_methods.py \
    --file1 submission.csv \
    --file2 submission_pure.csv
```

---

## 📊 예상 결과 비교 (Sample 1)

| Method | Result | 특징 |
|--------|--------|------|
| Leaf-centric | `[15, 17, 56]` | Leaf 보장, 경로 기반 |
| Pure Threshold | `[3, 17, 28]` | 최고 확률, 논문 충실 |

---

## 🎯 선택 가이드

### Leaf-centric 사용 시기
- ✅ 구체적 카테고리 중요
- ✅ Leaf 예측 필수
- ✅ Kaggle 등 실용적 목적

### Pure Threshold 사용 시기
- ✅ 논문 재현/비교
- ✅ 모델 확신 최대 활용
- ✅ Multi-label 완전 활용

---

## 📁 수정/생성된 파일

### 수정
1. **`generate_submission.py`**
   - `select_pure_threshold()` 함수 추가
   - `--pure_threshold` argument 추가
   - 방식 선택 로직 추가

### 생성
1. **`PREDICTION_ANALYSIS.md`** - 원인 분석 (상세)
2. **`METHOD_COMPARISON.md`** - 두 방식 비교 (심층)
3. **`INFERENCE_METHODS_GUIDE.md`** - 사용 가이드
4. **`compare_methods.py`** - 비교 도구
5. **`INFERENCE_UPDATE_SUMMARY.md`** - 이 문서

---

## 🔬 실험 권장사항

```bash
# 1. 두 방식 생성
python generate_submission.py --threshold 0.5 --output leaf.csv
python generate_submission.py --threshold 0.5 --pure_threshold --output pure.csv

# 2. 비교
python compare_methods.py --file1 leaf.csv --file2 pure.csv

# 3. Threshold 최적화
for t in 0.3 0.4 0.5 0.6 0.7; do
    python generate_submission.py --threshold $t --pure_threshold --output pure_t${t}.csv
done

# 4. 검증 데이터로 평가
# (최적 threshold 선택)

# 5. 최종 제출
python generate_submission.py --threshold 0.5 --pure_threshold --output final.csv
```

---

## 📚 참고 문서

- **PREDICTION_ANALYSIS.md** - 왜 [15,17,56]이 되었는가?
- **METHOD_COMPARISON.md** - Leaf-centric vs Pure Threshold
- **INFERENCE_METHODS_GUIDE.md** - 실행 가이드
- **INFERENCE_ANALYSIS.md** - 전체 inference 전략

---

## ✨ 주요 개선사항

1. ✅ **원인 파악**: Sample 1 불일치 이유 분석
2. ✅ **논문 방식 구현**: Pure threshold + ancestor closure
3. ✅ **선택 가능**: `--pure_threshold` flag로 방식 선택
4. ✅ **비교 도구**: `compare_methods.py`로 결과 비교
5. ✅ **문서화**: 4개 상세 문서 제공

---

## 🎉 결론

**문제**: Leaf-centric 후처리가 높은 확률 클래스를 무시

**해결**: Pure threshold 방식 추가로 논문에 충실한 선택지 제공

**결과**: 두 방식을 자유롭게 실험하고 최적 방식 선택 가능!

```bash
# 논문 재현
python generate_submission.py --threshold 0.5 --pure_threshold

# 실용적 목적
python generate_submission.py --threshold 0.5
```

이제 TaxoClass framework가 논문에 더 충실하면서도 유연한 inference를 지원합니다! 🚀
