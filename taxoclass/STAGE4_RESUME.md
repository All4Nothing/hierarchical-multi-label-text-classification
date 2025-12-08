# Stage 4 실행 가이드

## 변경 사항

### 1. `config.py` 수정
- `START_FROM_STAGE` 파라미터를 `4`로 설정
- 이제 파이프라인이 Stage 4 (Self-Training)부터 시작됩니다

### 2. `main.py` 수정
- Stage 3 섹션에 모델 로딩 로직 추가
- `START_FROM_STAGE > 3`이고 `best_model.pt`가 존재하면:
  - 학습을 건너뛰고 저장된 모델을 로드
  - Multi-GPU 설정 유지
  - Stage 4로 바로 진행

## 필요한 파일

다음 파일들이 존재해야 합니다 (✅ 모두 확인됨):

1. **Stage 1 출력**: `outputs/similarity_matrix_all.npz` (90MB)
2. **Stage 2 출력**: `outputs/core_classes.npz` (131KB)
3. **Stage 3 출력**: `saved_models/best_model.pt` (1.3GB)

## 실행 방법

```bash
cd /workspace/yongjoo/20252R0136DATA30400/taxoclass
python main.py
```

## 실행 순서

파이프라인은 다음과 같이 진행됩니다:

1. **Data Loading** - 데이터 및 계층 구조 로드
2. **Stage 1 (SKIPPED)** - `similarity_matrix_all.npz`에서 로드
3. **Stage 2 (SKIPPED)** - `core_classes.npz`에서 로드
4. **Stage 3 (SKIPPED)** - `best_model.pt`에서 모델 로드
5. **Stage 4 (RUN)** - Self-Training 실행
6. **Evaluation** - 테스트 세트 평가

## Stage 4 설정

`config.py`의 Self-Training 파라미터:
- `SELF_TRAIN_ITERATIONS = 3` - 3번의 self-training iteration
- `SELF_TRAIN_EPOCHS_PER_ITER = 3` - iteration당 3 epochs
- `SELF_TRAIN_TEMPERATURE = 2.0` - temperature scaling
- `SELF_TRAIN_THRESHOLD = 0.8` - 높은 신뢰도만 사용
- `SELF_TRAIN_LR = 1e-6` - 매우 낮은 learning rate

## 다시 처음부터 실행하려면

`config.py`에서:
```python
START_FROM_STAGE = None  # 또는 1
```

## 특정 Stage부터 실행하려면

- Stage 1부터: `START_FROM_STAGE = 1` 또는 `None`
- Stage 2부터: `START_FROM_STAGE = 2`
- Stage 3부터: `START_FROM_STAGE = 3`
- Stage 4부터: `START_FROM_STAGE = 4` ✅ (현재 설정)

## 주의사항

- 각 Stage를 건너뛰려면 해당 Stage의 출력 파일이 반드시 존재해야 합니다
- Stage 4를 실행하면 `saved_models/self_train_iter_X.pt` 형태로 새로운 모델이 저장됩니다
- Multi-GPU 환경에서도 정상적으로 작동합니다
