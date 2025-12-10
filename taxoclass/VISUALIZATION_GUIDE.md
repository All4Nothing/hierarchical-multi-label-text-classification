# Similarity Matrix Visualization Guide

## 📋 개요

`visualize_similarity.py`는 TaxoClass framework의 문서-클래스 유사도 행렬을 다양한 방식으로 시각화합니다.

### 지원하는 시각화

1. **Matrix Overview** - 전체 행렬 개요
2. **Class Statistics** - 클래스별 통계
3. **Document Statistics** - 문서별 통계
4. **Hierarchical Analysis** - 계층 구조 분석
5. **Top-K Analysis** - Top-K 클래스 분석

---

## 🚀 빠른 시작

### 모든 시각화 생성
```bash
python visualize_similarity.py --all
```

### 특정 시각화만 생성
```bash
# Matrix overview만
python visualize_similarity.py --overview

# Class statistics만
python visualize_similarity.py --class_stats

# Document statistics만
python visualize_similarity.py --doc_stats

# Hierarchical analysis만
python visualize_similarity.py --hierarchical
```

### 커스텀 설정
```bash
python visualize_similarity.py \
    --matrix_file outputs/similarity_matrix_all.npz \
    --output_dir outputs/visualizations \
    --all \
    --top_k 20
```

---

## 📊 시각화 상세 설명

### 1. Matrix Overview (`similarity_matrix_overview.png`)

**포함 내용**:
- 전체 행렬 히트맵 (샘플링된 버전)
- 유사도 점수 분포 히스토그램
- 문서별 최대 유사도 분포
- 클래스별 평균 유사도 분포

**용도**:
- 전체 데이터 분포 파악
- 이상치 탐지
- 데이터 품질 확인

**생성 명령어**:
```bash
python visualize_similarity.py --overview
```

---

### 2. Class Statistics (`class_statistics.png` + `class_statistics.csv`)

**포함 내용**:
- 평균 유사도 상위 20개 클래스
- 계층 레벨별 평균 유사도
- 평균 유사도 분포
- 최대 유사도 상위 20개 클래스
- 높은 유사도 문서 수 상위 20개 클래스
- 레벨별 유사도 분산

**용도**:
- 어떤 클래스가 가장 잘 매칭되는지 확인
- 계층 레벨별 패턴 분석
- 인기 클래스 식별

**생성 명령어**:
```bash
python visualize_similarity.py --class_stats
```

**CSV 파일**:
- 각 클래스의 상세 통계 저장
- 분석 및 추가 처리에 활용 가능

---

### 3. Document Statistics (`document_statistics.png`)

**포함 내용**:
- 문서별 최대 유사도 분포
- 문서별 평균 유사도 분포
- 문서별 높은 유사도 클래스 수 (threshold > 0.5)
- 최대 vs 평균 유사도 산점도

**용도**:
- 문서별 유사도 패턴 파악
- 난이도 높은 문서 식별
- 데이터 불균형 확인

**생성 명령어**:
```bash
python visualize_similarity.py --doc_stats
```

---

### 4. Hierarchical Analysis (`hierarchical_analysis.png`)

**포함 내용**:
- 레벨별 클래스 수 분포
- 레벨별 평균 유사도 (오차막대 포함)
- 레벨별 최대 유사도
- 레벨별 유사도 범위 (Min-Mean-Max)

**용도**:
- 계층 구조가 유사도에 미치는 영향 분석
- 레벨별 특성 파악
- 계층 설계 검증

**생성 명령어**:
```bash
python visualize_similarity.py --hierarchical
```

---

### 5. Top-K Analysis (`top_10_analysis.png`)

**포함 내용**:
- Top-K에 가장 자주 등장하는 클래스
- Top-K 빈도 분포
- Top-K 평균 유사도 분포
- Top-K 클래스 커버리지

**용도**:
- 인기 클래스 식별
- 다양성 분석
- Top-K 선택 전략 검증

**생성 명령어**:
```bash
python visualize_similarity.py --all --top_k 20
```

---

## 📁 출력 파일 구조

```
outputs/visualizations/
├── similarity_matrix_overview.png
├── class_statistics.png
├── class_statistics.csv
├── document_statistics.png
├── hierarchical_analysis.png
└── top_10_analysis.png
```

---

## 🔧 매개변수 설명

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--matrix_file` | `outputs/similarity_matrix_all.npz` | 유사도 행렬 파일 경로 |
| `--output_dir` | `outputs/visualizations` | 출력 디렉토리 |
| `--all` | False | 모든 시각화 생성 |
| `--overview` | False | Matrix overview만 생성 |
| `--class_stats` | False | Class statistics만 생성 |
| `--doc_stats` | False | Document statistics만 생성 |
| `--hierarchical` | False | Hierarchical analysis만 생성 |
| `--top_k` | 10 | Top-K 분석의 K 값 |

---

## 💡 사용 예시

### 예시 1: 전체 분석
```bash
python visualize_similarity.py --all
```

**결과**: 모든 시각화 생성

---

### 예시 2: 클래스 분석만
```bash
python visualize_similarity.py --class_stats
```

**결과**: 
- `class_statistics.png`
- `class_statistics.csv`

---

### 예시 3: Top-20 분석
```bash
python visualize_similarity.py --all --top_k 20
```

**결과**: Top-20 기준으로 분석

---

### 예시 4: 커스텀 경로
```bash
python visualize_similarity.py \
    --matrix_file custom/similarity.npz \
    --output_dir custom/viz \
    --all
```

---

## 📈 해석 가이드

### Matrix Overview 해석

**정상적인 패턴**:
- 유사도 분포가 정규분포에 가까움
- 문서별 최대 유사도가 0.5 이상
- 클래스별 평균이 균등하게 분포

**이상 패턴**:
- 유사도가 모두 매우 낮음 (< 0.1) → 모델 문제 가능
- 특정 클래스만 높은 유사도 → 데이터 불균형
- 문서별 최대 유사도가 매우 낮음 → 매칭 실패

---

### Class Statistics 해석

**유용한 인사이트**:
- **높은 평균 유사도**: 일반적인/포괄적인 클래스
- **높은 최대 유사도**: 특정 문서와 강한 매칭
- **높은 카운트**: 많은 문서와 관련

**활용**:
- Core class mining 검증
- 클래스 중요도 평가
- 데이터 불균형 확인

---

### Hierarchical Analysis 해석

**정상적인 패턴**:
- 상위 레벨(Level 0)이 높은 평균 유사도
- 하위 레벨로 갈수록 유사도 분산 증가
- 각 레벨에 적절한 클래스 수

**이상 패턴**:
- 특정 레벨만 유사도가 높음 → 계층 구조 문제
- 레벨별 차이가 없음 → 계층 정보 미활용

---

## 🔍 디버깅

### 파일 로드 오류
```
KeyError: 'similarity_matrix'
```

**해결**: 파일 내부 키 확인
```python
import numpy as np
data = np.load('outputs/similarity_matrix_all.npz')
print(data.keys())  # 사용 가능한 키 확인
```

### 메모리 부족
```
MemoryError: Unable to allocate array
```

**해결**: 
- 행렬이 너무 크면 자동 샘플링됨
- 더 작은 샘플 사용:
```python
# visualize_similarity.py에서 sample_size 조정
```

### 계층 정보 오류
```
KeyError: class_id not in hierarchy
```

**해결**: 
- `class_hierarchy.txt` 파일 확인
- 클래스 ID 범위 확인

---

## 📚 관련 파일

- `outputs/similarity_matrix_all.npz` - 입력 유사도 행렬
- `utils/hierarchy.py` - 계층 구조 관리
- `config.py` - 설정 파일
- `models/similarity.py` - 유사도 계산 모듈

---

## ✅ 체크리스트

실행 전:
- [ ] `similarity_matrix_all.npz` 파일 존재
- [ ] `class_hierarchy.txt` 파일 존재
- [ ] `classes.txt` 파일 존재
- [ ] 출력 디렉토리 생성 권한

실행 후:
- [ ] 모든 PNG 파일 생성 확인
- [ ] CSV 파일 생성 확인
- [ ] 이미지 품질 확인 (DPI 300)
- [ ] 통계 값이 합리적인지 확인

---

## 🎯 활용 시나리오

### 시나리오 1: Core Class Mining 검증
```bash
# 클래스 통계 확인
python visualize_similarity.py --class_stats

# 결과 확인:
# - 어떤 클래스가 자주 선택되는지
# - 평균 유사도가 높은 클래스
# - Core class mining 결과와 비교
```

### 시나리오 2: 모델 성능 분석
```bash
# 전체 분석
python visualize_similarity.py --all

# 확인 사항:
# - 유사도 분포가 정상인지
# - 특정 클래스에 편향이 있는지
# - 계층 구조가 잘 반영되는지
```

### 시나리오 3: 데이터 품질 검증
```bash
# 문서 통계 확인
python visualize_similarity.py --doc_stats

# 확인 사항:
# - 문서별 유사도가 적절한지
# - 너무 낮은 유사도 문서가 많은지
# - 데이터 불균형 여부
```

---

## 🎉 요약

```bash
# 한 번에 모든 시각화 생성
python visualize_similarity.py --all

# 결과 확인
ls -lh outputs/visualizations/
```

**주요 출력**:
- 5개 PNG 파일 (고해상도)
- 1개 CSV 파일 (상세 통계)

**활용**:
- 모델 성능 분석
- 데이터 품질 검증
- Core class mining 검증
- 계층 구조 분석

이제 유사도 행렬을 다양한 관점에서 분석할 수 있습니다! 📊
