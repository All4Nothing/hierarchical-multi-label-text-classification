# 에러 수정 및 추가 에러 가능성 분석

## 발생한 에러

```
ValueError: edge_index must be provided either as argument or registered buffer
```

### 원인
Stage 4 (Self-Training)를 시작할 때, 저장된 모델을 로드했지만 `edge_index`가 모델 버퍼로 등록되지 않았고, DataParallel 환경에서 제대로 전달되지 않음.

## 수정 사항

### 1. `main.py` - Stage 3 스킵 로직 수정

**문제점:**
- `edge_index`를 모델 state dict 로드 **후**에 등록
- DataParallel 래핑을 하지 않아 SelfTrainer와 불일치

**수정:**
```python
# BEFORE loading state dict
model.register_buffer('edge_index', edge_index)

# Load with strict=False (edge_index is new buffer)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# Move to device
model = model.to(main_device)

# Wrap with DataParallel if needed
if use_multi_gpu:
    model = torch.nn.DataParallel(model)
```

### 2. `self_training.py` - DataParallel 모델 처리

**문제점:**
- 이미 DataParallel로 래핑된 모델에 직접 버퍼 등록 시도
- 버퍼는 실제 모델(`model.module`)에 등록해야 함

**수정:**
```python
# Unwrap DataParallel to get actual model
actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model

# Register buffer on actual model
if not hasattr(actual_model, 'edge_index') or actual_model.edge_index is None:
    actual_model.register_buffer('edge_index', edge_index.to(device), persistent=False)
```

### 3. `metrics.py` - 평가 함수 수정

**문제점:**
- `evaluate_model`과 `predict_top_k_classes`도 동일한 DataParallel 이슈 존재
- `set_return_probs()`를 wrapper에서 호출하면 실제 모델에 전달 안 됨

**수정:**
```python
# Unwrap DataParallel
actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model

# Register buffer on actual model
actual_model.register_buffer('edge_index', edge_index.to(device), persistent=False)

# Set return_probs on actual model
actual_model.set_return_probs(True)
```

### 4. `main.py` - 모델 참조 업데이트

**문제점:**
- Self-training 후 모델이 SelfTrainer 내부에서 수정됨
- 평가 시 이전 모델 참조를 사용하면 업데이트되지 않은 모델로 평가

**수정:**
```python
# After self-training
model = self_trainer.model  # Update reference
```

## 추가 에러 가능성 분석

### 1. ⚠️ 메모리 부족 (OOM)
**발생 조건:**
- Stage 4에서 전체 문서(train + test = 49,145개)를 한번에 예측
- Multi-GPU 환경에서 모델 복제로 인한 추가 메모리 사용

**대응 방안:**
- `config.py`에서 `BATCH_SIZE` 감소 (현재 16)
- `EVAL_BATCH_SIZE` 감소 (현재 128)
- Gradient checkpointing 활성화

**예방 코드 (필요시 추가):**
```python
# config.py
SELF_TRAIN_BATCH_SIZE = 8  # Smaller batch for self-training
```

### 2. ⚠️ State Dict 키 불일치
**발생 조건:**
- 저장된 모델이 DataParallel로 저장되었는데 단일 GPU로 로드
- 또는 그 반대 경우

**현재 대응:**
- `strict=False`로 로드하여 누락/추가 키 허용
- 하지만 완전한 불일치는 감지 못함

**추가 예방 코드:**
```python
# Check if state dict has 'module.' prefix
state_dict = checkpoint['model_state_dict']
is_data_parallel_saved = any(k.startswith('module.') for k in state_dict.keys())
is_data_parallel_current = isinstance(model, torch.nn.DataParallel)

if is_data_parallel_saved and not is_data_parallel_current:
    # Remove 'module.' prefix
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
elif not is_data_parallel_saved and is_data_parallel_current:
    # Add 'module.' prefix
    state_dict = {f'module.{k}': v for k, v in state_dict.items()}
```

### 3. ⚠️ Gradient Overflow (Mixed Precision)
**발생 조건:**
- `USE_MIXED_PRECISION = True`인 상태에서 gradient가 너무 큼
- Self-training의 낮은 learning rate와 충돌 가능

**현재 대응:**
- Gradient clipping (`max_norm=1.0`)
- 매우 낮은 learning rate (`1e-6`)

**추가 모니터링:**
```python
# Check for NaN/Inf in losses
if torch.isnan(loss) or torch.isinf(loss):
    print("WARNING: NaN or Inf detected in loss!")
    continue  # Skip this batch
```

### 4. ⚠️ Tokenizer 불일치
**발생 조건:**
- Stage 3와 Stage 4에서 다른 tokenizer 설정 사용
- 저장된 모델이 다른 tokenizer로 학습됨

**현재 대응:**
- `DOC_MAX_LENGTH = 512`로 통일
- `BertTokenizer.from_pretrained(Config.DOC_ENCODER_MODEL)` 동일 사용

**확인 필요:**
- Stage 3 학습 시 사용한 설정과 동일한지 확인

### 5. ⚠️ CUDA Out of Memory in DataParallel
**발생 조건:**
- DataParallel의 불균형한 메모리 분배
- 첫 번째 GPU에 더 많은 메모리 사용

**현재 대응:**
- Batch size를 작게 유지

**개선 방안:**
- DistributedDataParallel (DDP) 사용 (더 효율적)
- 하지만 코드 복잡도 증가

### 6. ⚠️ Wandb 연결 실패
**발생 조건:**
- 네트워크 문제로 wandb 로깅 실패
- 학습은 계속되지만 모니터링 불가

**현재 대응:**
- `try-except`로 wandb import 처리
- `use_wandb` 플래그로 선택적 사용

**안정성:**
- 이미 잘 처리되어 있음

### 7. ⚠️ 파일 저장 실패
**발생 조건:**
- 디스크 용량 부족
- 권한 문제

**현재 대응:**
- 디렉토리 생성 확인 (`os.makedirs(..., exist_ok=True)`)

**추가 확인:**
```python
# Check disk space before saving
import shutil
disk_usage = shutil.disk_usage(Config.MODEL_SAVE_DIR)
free_gb = disk_usage.free / (1024**3)
if free_gb < 5.0:  # Less than 5GB
    print(f"WARNING: Low disk space ({free_gb:.2f} GB)")
```

## 테스트 체크리스트

실행 전 확인사항:

1. ✅ `config.py`에서 `START_FROM_STAGE = 4` 설정
2. ✅ `saved_models/best_model.pt` 존재 확인 (1.3GB)
3. ✅ `outputs/similarity_matrix_all.npz` 존재 확인 (90MB)
4. ✅ `outputs/core_classes.npz` 존재 확인 (131KB)
5. ⚠️ GPU 메모리 확인 (`nvidia-smi`)
6. ⚠️ 디스크 공간 확인 (`df -h`)

## 실행 명령어

```bash
cd /workspace/yongjoo/20252R0136DATA30400/taxoclass
python main.py
```

## 예상 실행 시간

- Stage 4 각 iteration당:
  - 예측 생성: ~10-15분 (49,145 documents)
  - 학습 (3 epochs): ~30-45분
- 총 예상 시간: **2-3시간** (3 iterations)

## 모니터링

실행 중 확인사항:
1. "Model loaded and ready for Stage 4" 메시지 확인
2. Self-training iteration별 loss 감소 확인
3. Confident predictions 비율 확인 (threshold 0.8)
4. GPU 메모리 사용량 모니터링

## 문제 발생 시

1. **OOM 에러**: `config.py`에서 batch size 감소
2. **Loss가 NaN**: Mixed precision 비활성화
3. **매우 느림**: Multi-GPU 정상 작동 확인
4. **정확도 하락**: Learning rate 더 낮추기
