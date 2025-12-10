# Multi-GPU TELEClass Pipeline Guide

## Overview

`pipeline_teleclass_multigpu.py`는 여러 GPU를 활용하여 TELEClass 파이프라인을 **2-4배 빠르게** 실행할 수 있는 최적화된 버전입니다.

## 주요 개선사항

### 🚀 성능 향상

1. **Multi-GPU 문서 인코딩** (Phase 1)
   - SentenceTransformer가 모든 가용 GPU에 자동으로 워크로드 분산
   - 배치 크기가 GPU 수에 비례하여 자동 증가
   - 예: 4 GPU × 64 batch = 256 effective batch size

2. **DataParallel BERT 학습** (Phase 5)
   - 여러 GPU에 걸쳐 배치를 자동으로 분할
   - 각 GPU가 독립적으로 forward/backward pass 수행
   - 그래디언트 자동 집계 및 동기화

3. **Mixed Precision Training (FP16)**
   - 메모리 사용량 ~50% 감소
   - 학습 속도 ~30-50% 향상
   - NVIDIA Tensor Core 활용

4. **Multi-GPU 추론** (Phase 6)
   - 테스트 데이터를 여러 GPU에 분산
   - 배치 크기 자동 스케일링
   - 병렬 예측으로 추론 시간 단축

### 🎯 자동 최적화

- **자동 GPU 감지**: 사용 가능한 모든 GPU 자동 탐지
- **동적 배치 크기**: GPU 수에 따라 자동 조정
- **메모리 최적화**: Pin memory 및 효율적인 데이터 로딩
- **워커 프로세스**: 멀티프로세싱으로 I/O 병목 해소

## 시스템 요구사항

### 하드웨어
- **GPU**: NVIDIA GPU 2개 이상 권장 (1개도 가능)
- **VRAM**: GPU당 최소 16GB 권장
- **RAM**: 32GB+ 권장
- **CUDA**: 11.0 이상

### 소프트웨어
```bash
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
pandas>=1.5.0
numpy>=1.23.0
networkx>=3.0
scikit-learn>=1.2.0
tqdm>=4.65.0
```

## 설치 및 실행

### 1. 의존성 설치

```bash
cd modified_teleclass
pip install -r requirements_teleclass.txt
```

### 2. GPU 확인

```python
import torch
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
```

예상 출력:
```
Available GPUs: 4
  GPU 0: NVIDIA RTX 6000 Ada Generation
  GPU 1: NVIDIA RTX 6000 Ada Generation
  GPU 2: NVIDIA RTX 6000 Ada Generation
  GPU 3: NVIDIA RTX 6000 Ada Generation
```

### 3. 파이프라인 실행

#### 방법 1: 모든 GPU 자동 사용
```bash
python pipeline_teleclass_multigpu.py
```

#### 방법 2: 특정 GPU 지정
```python
from pipeline_teleclass_multigpu import MultiGPUTELEClassPipeline

# GPU 0, 1, 2만 사용
pipeline = MultiGPUTELEClassPipeline(
    data_dir="../Amazon_products",
    output_dir="outputs",
    seed=42,
    device_ids=[0, 1, 2]  # 원하는 GPU ID 지정
)
pipeline.run()
```

#### 방법 3: 환경 변수로 GPU 제한
```bash
# GPU 0과 1만 사용
CUDA_VISIBLE_DEVICES=0,1 python pipeline_teleclass_multigpu.py

# GPU 2와 3만 사용
CUDA_VISIBLE_DEVICES=2,3 python pipeline_teleclass_multigpu.py
```

## 성능 비교

### 예상 실행 시간 (4x NVIDIA RTX 6000 Ada)

| Phase | Single GPU | 4 GPU Multi-GPU | 속도 향상 |
|-------|-----------|-----------------|----------|
| Phase 1: Encoding | 8-10 min | 2-3 min | **~3-4x** |
| Phase 2: Refinement | 3-5 min | 3-5 min | ~1x (GPU 간 통신 오버헤드) |
| Phase 3: Augmentation | 1 min | 1 min | ~1x |
| Phase 4: Hierarchy | 1 min | 1 min | ~1x |
| Phase 5: BERT Training | 50-60 min | 15-20 min | **~3-4x** |
| Phase 6: Inference | 8-10 min | 2-3 min | **~3-4x** |
| **Total** | **70-90 min** | **25-35 min** | **~2.5-3x** |

### 메모리 사용량

| 설정 | GPU당 VRAM 사용량 |
|------|------------------|
| Single GPU (FP32) | ~14-16 GB |
| Multi-GPU (FP32) | ~10-12 GB |
| Multi-GPU (FP16) | ~6-8 GB |

## 주요 기능 설명

### 1. MultiGPUClassRepresentation

**문서 인코딩 병렬화:**
```python
class_repr = MultiGPUClassRepresentation(device_ids=[0, 1, 2, 3])

# 자동으로 4개 GPU에 분산
doc_embeddings = class_repr.encode_documents_parallel(
    all_corpus,
    batch_size=64  # GPU당 64, 총 256 effective batch
)
```

**작동 방식:**
- SentenceTransformer가 내부적으로 DataParallel 사용
- 각 GPU가 배치의 일부를 처리
- 결과를 자동으로 집계

### 2. MultiGPUBERTTrainer

**DataParallel 학습:**
```python
trainer = MultiGPUBERTTrainer(
    num_classes=531,
    device_ids=[0, 1, 2, 3],
    use_mixed_precision=True  # FP16 활성화
)

trainer.prepare_data(
    train_texts, 
    train_labels, 
    batch_size=16  # GPU당 16, 총 64 effective batch
)

trainer.train(num_epochs=3)
```

**작동 방식:**
1. 모델이 DataParallel로 래핑됨
2. 각 배치가 GPU 수만큼 분할
3. 각 GPU가 forward pass 수행
4. Loss가 primary GPU에서 집계
5. Backward pass 후 그래디언트 동기화
6. Optimizer step 수행

**Mixed Precision Training:**
```python
# FP16 연산으로 메모리 절약 및 속도 향상
with autocast():
    outputs = model(input_ids, attention_mask)
    loss = criterion(outputs.logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. MultiGPUInference

**병렬 추론:**
```python
inference = MultiGPUInference(
    model_path="outputs/models/best_model",
    device_ids=[0, 1, 2, 3]
)

# 자동으로 4개 GPU에 분산
predictions = inference.predict(
    test_corpus,
    batch_size=32  # GPU당 32, 총 128 effective batch
)
```

## 고급 설정

### 1. 배치 크기 튜닝

```python
# 메모리 부족 시 감소
trainer.prepare_data(train_texts, train_labels, batch_size=8)

# 메모리 여유 시 증가
trainer.prepare_data(train_texts, train_labels, batch_size=32)
```

**권장 배치 크기:**
- VRAM 16GB: batch_size=16
- VRAM 24GB: batch_size=24-32
- VRAM 48GB: batch_size=32-48

### 2. 워커 프로세스 수 조정

```python
# CPU 코어 수에 따라 조정
trainer.prepare_data(
    train_texts, 
    train_labels, 
    num_workers=8  # CPU 코어 수의 50-75%
)
```

### 3. Mixed Precision 비활성화

```python
# 정확도가 중요한 경우
trainer = MultiGPUBERTTrainer(
    num_classes=531,
    use_mixed_precision=False  # FP32 사용
)
```

### 4. 특정 Phase만 Multi-GPU 사용

```python
# Phase 1: Multi-GPU 인코딩만 사용
class_repr = MultiGPUClassRepresentation(device_ids=[0, 1])
doc_embeddings = class_repr.encode_documents_parallel(all_corpus)

# Phase 5: Single GPU 학습 (메모리 부족 시)
trainer = MultiGPUBERTTrainer(device_ids=[0])
```

## 트러블슈팅

### Issue 1: CUDA Out of Memory

**증상:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**해결책:**
```python
# 1. 배치 크기 감소
trainer.prepare_data(train_texts, train_labels, batch_size=8)

# 2. Mixed Precision 활성화
use_mixed_precision=True

# 3. 사용 GPU 수 감소
device_ids=[0, 1]  # 4개 대신 2개만 사용
```

### Issue 2: GPU 간 성능 불균형

**증상:**
```
GPU 0: 95% utilization
GPU 1: 30% utilization
GPU 2: 25% utilization
GPU 3: 20% utilization
```

**해결책:**
```python
# DataParallel 대신 DistributedDataParallel 고려 (향후 구현)
# 현재는 배치 크기를 조정하여 완화
batch_size=16  # 더 큰 배치로 균등 분산
```

### Issue 3: 느린 데이터 로딩

**증상:**
```
GPU utilization: 30-40% (should be 80-100%)
```

**해결책:**
```python
# 워커 수 증가
num_workers=8  # 기본값 4에서 증가

# Pin memory 활성화 (자동 활성화됨)
pin_memory=True
```

### Issue 4: Multi-GPU에서 성능 향상 없음

**점검 사항:**
1. GPU 간 통신 대역폭 확인:
   ```bash
   nvidia-smi topo -m
   ```

2. PCIe 연결 상태 확인:
   ```bash
   nvidia-smi
   # Link: P2P 또는 SYS 확인
   ```

3. 배치 크기가 충분히 큰지 확인:
   ```python
   # 너무 작은 배치는 오버헤드 증가
   batch_size=16  # 최소 권장값
   ```

## 모니터링

### 실시간 GPU 사용률 모니터링

```bash
# 터미널 1: 파이프라인 실행
python pipeline_teleclass_multigpu.py

# 터미널 2: GPU 모니터링
watch -n 1 nvidia-smi
```

### Python 코드로 모니터링

```python
import subprocess
import time

def monitor_gpus():
    while True:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
        time.sleep(1)

# 별도 스레드에서 실행
```

## 성능 최적화 팁

### 1. 최적의 GPU 수 선택

- **2 GPU**: 가장 효율적 (통신 오버헤드 최소)
- **4 GPU**: 균형잡힌 성능 (권장)
- **8+ GPU**: 추가 이득 제한적 (통신 오버헤드 증가)

### 2. 배치 크기 최적화

```python
# GPU 수 × 16 또는 32가 일반적으로 최적
# 4 GPU: batch_size=16 → effective=64
# 4 GPU: batch_size=32 → effective=128
```

### 3. 데이터 로딩 최적화

```python
num_workers = min(8, os.cpu_count() // 2)
pin_memory = True  # GPU 전송 속도 향상
```

### 4. Gradient Accumulation (대체 방법)

메모리가 부족하지만 큰 effective batch를 원할 때:

```python
# 향후 추가 구현 예정
accumulation_steps = 4
effective_batch = batch_size * accumulation_steps * num_gpus
```

## 비교: Single vs Multi-GPU

### 언제 Multi-GPU를 사용해야 할까?

**Multi-GPU 사용 권장:**
- ✅ GPU가 2개 이상 있을 때
- ✅ 빠른 실험 반복이 필요할 때
- ✅ 큰 배치 크기가 필요할 때
- ✅ 문서 수가 많을 때 (50K+)

**Single GPU 사용 권장:**
- ✅ GPU가 1개만 있을 때
- ✅ 메모리가 충분할 때
- ✅ 작은 데이터셋 (<10K 문서)
- ✅ 디버깅 시

### 코드 차이

```python
# Single GPU
from pipeline_teleclass import TELEClassPipeline
pipeline = TELEClassPipeline(data_dir="../Amazon_products")
pipeline.run()

# Multi-GPU
from pipeline_teleclass_multigpu import MultiGPUTELEClassPipeline
pipeline = MultiGPUTELEClassPipeline(
    data_dir="../Amazon_products",
    device_ids=[0, 1, 2, 3]  # 또는 None으로 자동
)
pipeline.run()
```

## 추가 개선 사항 (향후)

현재 구현에서 더 개선할 수 있는 부분:

1. **DistributedDataParallel (DDP)**
   - DataParallel보다 더 효율적
   - GPU 간 통신 최적화
   - 더 나은 확장성

2. **Gradient Accumulation**
   - 작은 GPU 메모리에서 큰 effective batch
   - 메모리 효율성 증가

3. **Model Parallelism**
   - 매우 큰 모델을 GPU 간 분할
   - 더 큰 모델 사용 가능

4. **Pipeline Parallelism**
   - 레이어를 GPU 간 분할
   - 지속적인 GPU 활용

## 참고 자료

- [PyTorch DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)
- [PyTorch Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [SentenceTransformers Multi-GPU](https://www.sbert.net/docs/training/overview.html#multi-gpu-training)

## 요약

| 특징 | Single GPU | Multi-GPU |
|------|-----------|-----------|
| 실행 시간 | 70-90 분 | 25-35 분 |
| GPU 메모리 | 14-16 GB | 6-8 GB (FP16) |
| 설정 복잡도 | ⭐ 간단 | ⭐⭐ 보통 |
| 디버깅 | ⭐⭐⭐ 쉬움 | ⭐⭐ 보통 |
| 확장성 | 제한적 | 우수 |
| 권장 사용 | 프로토타입, 디버깅 | 프로덕션, 대규모 |

---

**Quick Start:**
```bash
cd modified_teleclass
python pipeline_teleclass_multigpu.py
```

**Expected:** ~25-35분 소요, 4 GPU 사용 시 ~3배 속도 향상! 🚀
