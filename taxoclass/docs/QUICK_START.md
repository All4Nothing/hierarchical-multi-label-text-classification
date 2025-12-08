# 🚀 TaxoClass 빠른 시작 가이드

## 📦 설치

### 1. 필수 패키지 설치

```bash
cd taxoclass
pip install -r requirements.txt
```

### 2. 설치 확인

```bash
python quick_test.py
```

## ⚙️ 설정

### 데이터 경로 설정

`config.py` 파일을 열고 데이터 경로를 확인/수정하세요:

```python
DATA_DIR = "../Amazon_products"
```

현재 설정:
- Classes: `Amazon_products/classes.txt`
- Hierarchy: `Amazon_products/class_hierarchy.txt`
- Train: `Amazon_products/train/train_corpus.txt`
- Test: `Amazon_products/test/test_corpus.txt`

## 🏃 실행

### 방법 1: 전체 파이프라인 실행

```bash
python main.py
```

또는:

```bash
./run.sh
```

### 방법 2: 단계별 실행

```bash
# 예제 코드 실행 (계층 구조 탐색)
python example_usage.py
```

## 📊 예상 실행 시간

| 단계 | 시간 (CPU) | 시간 (GPU) |
|------|-----------|-----------|
| Stage 1: Similarity | ~2시간 | ~30분 |
| Stage 2: Core Mining | ~10분 | ~10분 |
| Stage 3: Training | ~4시간 | ~1시간 |
| Stage 4: Self-Training | ~6시간 | ~1.5시간 |
| **총합** | **~12시간** | **~3시간** |

*약 29,000개 문서, 532개 클래스 기준

## ⚡ 빠른 테스트 (작은 데이터셋)

빠르게 테스트하려면 `config.py`를 수정하세요:

```python
# 훈련 에포크 줄이기
NUM_EPOCHS = 3

# 배치 사이즈 늘리기 (GPU 메모리가 충분한 경우)
BATCH_SIZE = 64

# Self-training 반복 줄이기
SELF_TRAIN_ITERATIONS = 2
```

또는 `main.py`에서:

```python
# Self-training 건너뛰기
run_self_training = False

# 빠른 유사도 계산 사용
use_fast_similarity = True
```

## 📁 출력 파일

실행 후 생성되는 파일들:

```
taxoclass/
├── cache/
│   └── similarity_matrix_*.pkl      # 캐시된 유사도 행렬
├── saved_models/
│   ├── best_model.pt                # 최고 성능 모델
│   ├── checkpoint_epoch_*.pt        # 체크포인트
│   └── self_train_iter_*.pt         # Self-training 모델
└── outputs/
    └── metrics.txt                   # 평가 결과
```

## 🐛 문제 해결

### CUDA Out of Memory

```python
# config.py 수정
BATCH_SIZE = 16
SIMILARITY_BATCH_SIZE = 8
```

### 느린 실행 속도

```python
# main.py 수정
use_fast_similarity = True  # 빠른 유사도 계산 사용
run_self_training = False   # Self-training 건너뛰기
```

### 데이터 파일을 찾을 수 없음

```bash
# 데이터 경로 확인
ls ../Amazon_products/

# config.py에서 DATA_DIR 경로 수정
```

## 📈 결과 확인

### 1. 콘솔 출력

실행 중 각 단계의 진행 상황과 결과가 출력됩니다.

### 2. 메트릭 파일

```bash
cat outputs/metrics.txt
```

### 3. 저장된 모델 사용

```python
from models.classifier import TaxoClassifier
import torch

model = TaxoClassifier(num_classes=532)
checkpoint = torch.load("saved_models/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
```

## 💡 팁

### GPU 메모리 최적화

1. **그래디언트 체크포인팅 활성화**
2. **Mixed Precision Training 사용**
3. **배치 사이즈 줄이기**

### 성능 향상

1. **더 많은 에포크 훈련**
2. **Learning rate 튜닝**
3. **GNN 레이어 수 증가**

### 디버깅

```python
# config.py
DEVICE = "cpu"  # CPU로 테스트
NUM_EPOCHS = 1  # 빠른 테스트
```

## 📞 도움말

문제가 발생하면:

1. `quick_test.py` 실행하여 설치 확인
2. `example_usage.py` 실행하여 기본 기능 테스트
3. 에러 메시지 확인
4. `config.py` 설정 확인

## 🎯 다음 단계

1. ✅ 설치 완료
2. ✅ 데이터 확인
3. ✅ 기본 실행
4. 📊 결과 분석
5. 🔧 하이퍼파라미터 튜닝
6. 🚀 프로덕션 배포

---

**Happy Classifying! 🎉**

