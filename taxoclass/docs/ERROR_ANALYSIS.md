# Stage 4 에러 분석 및 수정

## ✅ 수정 완료: edge_index 에러

### 발생한 에러
```
ValueError: edge_index must be provided either as argument or registered buffer
```

### 원인
- Stage 3을 건너뛰고 모델을 로드할 때, `edge_index`가 모델의 버퍼로 등록되지 않음
- DataParallel 환경에서 `edge_index` 없이 forward pass를 시도하여 실패

### 해결 방법
**main.py 수정사항:**
1. Stage 3을 건너뛸 때 모델 로드 후 `edge_index`를 버퍼로 등록
2. `model.register_buffer('edge_index', edge_index)` 추가
3. DataParallel 래핑은 SelfTrainer에게 위임 (이중 래핑 방지)

```python
# main.py Line ~425
model.register_buffer('edge_index', edge_index)ㅇ
model = model.to(main_device)
# DataParallel 래핑은 SelfTrainer에서 처리
```

---

## ⚠️ 추가 발생 가능한 에러 분석

### 1. 메모리 부족 에러 (OOM - Out of Memory)

**발생 가능 상황:**
- Self-training 시 전체 데이터셋(49,145개 문서)에 대해 prediction 생성
- Unlabeled dataset이 너무 크면 메모리 부족 발생 가능

**증상:**
```
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError
```

**예방 조치:**
- `config.py`에서 batch size 조정:
  ```python
  BATCH_SIZE = 16  # 메모리 부족 시 8로 감소
  EVAL_BATCH_SIZE = 128  # 메모리 부족 시 64로 감소
  ```
- Gradient accumulation 사용 (이미 설정됨):
  ```python
  GRADIENT_ACCUMULATION_STEPS = 4
  ```

**발생 시 대처:**
1. Batch size를 줄이기: `BATCH_SIZE = 8`, `EVAL_BATCH_SIZE = 64`
2. Mixed precision 활성화 확인: `USE_MIXED_PRECISION = True` (이미 설정됨)
3. GPU 캐시 정리: `torch.cuda.empty_cache()` 호출

---

### 2. DataParallel과 관련된 에러

**발생 가능 상황:**
- 모델이 이미 DataParallel로 래핑되어 있는데 다시 래핑하려고 할 때
- DataParallel 모델에서 module 접근 시 에러

**증상:**
```
AttributeError: 'DataParallel' object has no attribute 'xxx'
RuntimeError: module must have its parameters and buffers on device cuda:0
```

**예방 조치 (이미 적용됨):**
```python
# SelfTrainer에서 이중 래핑 방지
if use_multi_gpu and torch.cuda.device_count() > 1 and not isinstance(self.model, torch.nn.DataParallel):
    self.model = torch.nn.DataParallel(self.model)

# 실제 모델에 접근할 때
actual_model = self.model.module if hasattr(self.model, 'module') else self.model
```

**발생 시 대처:**
- 모델 저장 시: `model.module.state_dict()` 사용
- 모델 로드 시: unwrapped model에 로드

---

### 3. Target Distribution 관련 수치 에러

**발생 가능 상황:**
- Temperature sharpening 시 수치가 너무 작거나 커질 때
- Log 계산 시 0이 입력되어 `-inf` 발생

**증상:**
```
RuntimeError: Function 'LogBackward' returned nan values
RuntimeError: CUDA error: device-side assert triggered
```

**예방 조치 (이미 적용됨):**
```python
# self_training.py에서
eps = 1e-10  # Small epsilon for numerical stability
log_predictions = torch.log(predictions + eps)
log_target = torch.log(target_distribution + eps)
```

**설정 조정 가능:**
```python
# config.py
SELF_TRAIN_TEMPERATURE = 2.0  # 너무 작으면 (< 1) 불안정
SELF_TRAIN_THRESHOLD = 0.8  # 너무 높으면 학습 데이터 부족
```

---

### 4. 학습 중 Gradient Explosion/Vanishing

**발생 가능 상황:**
- Learning rate가 너무 높을 때
- Gradient가 폭발하거나 소실될 때

**증상:**
```
RuntimeError: Function 'MulBackward0' returned nan values
Loss becomes nan or inf
```

**예방 조치 (이미 적용됨):**
```python
# config.py
SELF_TRAIN_LR = 1e-6  # 매우 낮은 learning rate

# self_training.py
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

**발생 시 대처:**
1. Learning rate 더 낮추기: `SELF_TRAIN_LR = 5e-7`
2. Gradient clipping norm 낮추기: `max_norm=0.5`

---

### 5. Checkpoint 저장/로드 에러

**발생 가능 상황:**
- Self-training 도중 모델 저장 시 디스크 공간 부족
- Checkpoint 형식 불일치

**증상:**
```
OSError: [Errno 28] No space left on device
RuntimeError: Error(s) in loading state_dict
```

**예방 조치:**
- 디스크 공간 확인:
  ```bash
  df -h /workspace/yongjoo/20252R0136DATA30400/taxoclass/saved_models/
  ```
- 모델 크기 확인: ~1.3GB per checkpoint

**발생 시 대처:**
1. 이전 checkpoint 삭제
2. 저장 빈도 줄이기 (iteration마다만 저장)

---

### 6. Multi-label Prediction 관련 에러

**발생 가능 상황:**
- Prediction shape이 예상과 다를 때
- Target distribution shape mismatch

**증상:**
```
RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)
IndexError: index X is out of bounds for dimension Y
```

**예방 조치 (코드 검증):**
```python
# 데이터 shape 확인
print(f"Predictions shape: {predictions.shape}")  # Should be (49145, num_classes)
print(f"Target distribution shape: {target_distribution.shape}")
```

**발생 시 대처:**
- `num_classes` 값이 일치하는지 확인
- Hierarchy에서 계산된 class 수와 모델의 출력 차원 비교

---

### 7. Wandb 로깅 에러

**발생 가능 상황:**
- Wandb 인증 만료
- 네트워크 연결 문제

**증상:**
```
wandb.Error: api_key not configured
requests.exceptions.ConnectionError
```

**예방 조치:**
```python
# config.py에서 wandb 비활성화 가능
USE_WANDB = False  # 에러 발생 시 False로 변경
```

**발생 시 대처:**
1. Wandb 로그인 다시 실행: `wandb login`
2. 또는 wandb 비활성화하고 계속 진행

---

## 🔍 모니터링 체크리스트

실행 중 다음 사항들을 모니터링하세요:

### 메모리 사용량
```bash
watch -n 1 nvidia-smi
```

### Loss 값
- Loss가 `nan`이나 `inf`가 되지 않는지 확인
- Loss가 너무 빠르게 증가하면 learning rate 조정 필요

### Confidence 비율
- Self-training iteration마다 출력되는 confidence ratio 확인
- 너무 낮으면 (<10%) threshold 조정 필요

### Disk 공간
```bash
df -h
```

---

## 📋 권장 실행 순서

1. **실행 전 확인:**
   ```bash
   # GPU 메모리 확인
   nvidia-smi
   
   # 디스크 공간 확인
   df -h
   
   # 필요한 파일 존재 확인
   ls -lh saved_models/best_model.pt
   ls -lh outputs/similarity_matrix_all.npz
   ls -lh outputs/core_classes.npz
   ```

2. **실행:**
   ```bash
   cd /workspace/yongjoo/20252R0136DATA30400/taxoclass
   python main.py
   ```

3. **실행 중 모니터링:**
   - 별도 터미널에서 `watch -n 1 nvidia-smi` 실행
   - Loss 값이 정상적으로 감소하는지 확인
   - Confidence ratio가 합리적인지 확인 (20-60% 정도 예상)

4. **에러 발생 시:**
   - 에러 메시지 전체를 복사
   - 위 에러 분석 섹션에서 해당 에러 찾기
   - 제안된 대처 방법 적용

---

## ⚙️ 긴급 설정 조정

메모리나 성능 문제 발생 시 `config.py`에서 다음 값들을 조정:

```python
# 메모리 부족 시
BATCH_SIZE = 8  # 16 -> 8
EVAL_BATCH_SIZE = 64  # 128 -> 64

# 학습이 불안정할 시
SELF_TRAIN_LR = 5e-7  # 1e-6 -> 5e-7
SELF_TRAIN_THRESHOLD = 0.7  # 0.8 -> 0.7

# 학습 데이터 부족 시
SELF_TRAIN_THRESHOLD = 0.6  # 0.8 -> 0.6 (더 많은 샘플 사용)

# 빠른 테스트용
SELF_TRAIN_ITERATIONS = 1  # 3 -> 1
SELF_TRAIN_EPOCHS_PER_ITER = 1  # 3 -> 1
```
