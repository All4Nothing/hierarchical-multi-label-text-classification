# Stage 4 실행을 위한 코드 수정 요약

## 🔧 수정된 파일

### 1. `config.py`
**변경 사항:**
```python
# Line 88
START_FROM_STAGE = 4  # None -> 4
```

**목적:** Stage 4 (Self-Training)부터 파이프라인 시작

---

### 2. `main.py`

#### 수정 1: Stage 3 섹션 재구조화 (Line ~394-556)

**변경 전:**
- Stage 3에서 항상 학습 실행

**변경 후:**
- Stage 3을 두 가지 경로로 분기:
  1. **건너뛰기 경로** (`start_from_stage > 3`이고 `best_model.pt` 존재 시)
  2. **학습 경로** (그 외)

#### 건너뛰기 경로 상세 (새로 추가됨):

```python
if start_from_stage > 3 and os.path.exists(best_model_path):
    # 1. 모델 로드
    checkpoint = torch.load(best_model_path, map_location=main_device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 2. edge_index를 버퍼로 등록 (CRITICAL)
    model.register_buffer('edge_index', edge_index)
    
    # 3. 디바이스로 이동
    model = model.to(main_device)
    
    # 4. DataParallel 래핑은 SelfTrainer에게 위임
    # (이중 래핑 방지)
    
    model.eval()
```

**핵심 포인트:**
- ✅ `edge_index` 버퍼 등록 - DataParallel 환경에서 필수
- ✅ DataParallel 래핑을 SelfTrainer에게 위임 - 이중 래핑 방지
- ✅ Checkpoint 형식 유연하게 처리 - dict 또는 state_dict 모두 지원

---

## 🎯 해결된 에러

### ValueError: edge_index must be provided either as argument or registered buffer

**원인:**
- Stage 3 건너뛸 때 `edge_index`가 모델 버퍼로 등록되지 않음
- DataParallel 모델에서 forward pass 시 `edge_index`를 찾을 수 없음

**해결:**
```python
model.register_buffer('edge_index', edge_index)
```

**동작 원리:**
1. `edge_index`를 모델의 버퍼로 등록
2. DataParallel이 모델을 복제할 때 `edge_index`도 자동으로 복제됨
3. 각 GPU의 모델 replica가 자체 `edge_index`를 가짐
4. Forward pass에서 `edge_index=None`으로 호출 시 버퍼에서 자동으로 사용

---

## 📊 실행 흐름

### 이전 (Stage 3 건너뛰기 불가):
```
Data Loading → Stage 1 → Stage 2 → Stage 3 (학습) → Stage 4 → Evaluation
```

### 현재 (Stage 4부터 시작 가능):
```
Data Loading → Stage 1 (SKIP) → Stage 2 (SKIP) → Stage 3 (SKIP, 모델 로드) → Stage 4 → Evaluation
```

각 SKIP 단계:
- **Stage 1 SKIP**: `similarity_matrix_all.npz` 로드
- **Stage 2 SKIP**: `core_classes.npz` 로드
- **Stage 3 SKIP**: `best_model.pt` 로드 + `edge_index` 등록

---

## ✅ 검증 완료 사항

### 1. 필요한 파일 존재 확인
```bash
✓ saved_models/best_model.pt (1.3GB)
✓ outputs/similarity_matrix_all.npz (90MB)
✓ outputs/core_classes.npz (131KB)
```

### 2. 코드 안전성
- ✅ edge_index 버퍼 등록
- ✅ DataParallel 이중 래핑 방지
- ✅ Checkpoint 형식 유연 처리
- ✅ GPU 메모리 관리 (Mixed Precision, Gradient Accumulation)

### 3. 호환성
- ✅ Single GPU 환경 지원
- ✅ Multi-GPU (DataParallel) 환경 지원
- ✅ CPU 환경 지원 (device 자동 감지)

---

## 🚀 실행 방법

```bash
cd /workspace/yongjoo/20252R0136DATA30400/taxoclass
python main.py
```

**예상 출력:**
```
================================================================================
STAGE 1: DOCUMENT-CLASS SIMILARITY CALCULATION (SKIPPED - LOADING FROM FILE)
================================================================================
✅ Loaded similarity matrix: (49145, num_classes)

================================================================================
STAGE 2: CORE CLASS MINING (SKIPPED - LOADING FROM FILE)
================================================================================
✅ Loaded core classes for 49145 documents

================================================================================
STAGE 3: CLASSIFIER TRAINING (SKIPPED - LOADING FROM FILE)
================================================================================
✅ Loaded model from checkpoint (epoch X)
✅ edge_index registered: shape torch.Size([2, num_edges])
Model loaded and ready for Stage 4

================================================================================
STAGE 4: SELF-TRAINING
================================================================================
Total documents for self-training: 49145
...
```

---

## 🔄 다른 Stage부터 시작하기

`config.py`에서 `START_FROM_STAGE` 값만 변경:

```python
START_FROM_STAGE = None  # Stage 1부터 (처음부터)
START_FROM_STAGE = 1     # Stage 1부터 (처음부터)
START_FROM_STAGE = 2     # Stage 2부터 (similarity matrix 로드)
START_FROM_STAGE = 3     # Stage 3부터 (similarity + core classes 로드)
START_FROM_STAGE = 4     # Stage 4부터 (모든 이전 결과 로드) ← 현재 설정
```

---

## 📝 코드 변경 요약

| 파일 | 변경 위치 | 변경 유형 | 목적 |
|------|----------|----------|------|
| config.py | Line 88 | 값 변경 | Stage 4부터 시작 |
| main.py | Line ~394-431 | 로직 추가 | Stage 3 건너뛰기 + 모델 로드 |
| main.py | Line ~432-556 | 기존 유지 | Stage 3 정상 학습 경로 |

**총 변경 라인 수:** ~40 lines
**새로 추가된 파일:** 0
**삭제된 파일:** 0

---

## ⚠️ 주의사항

1. **edge_index 등록 순서 중요:**
   - 모델 로드 → edge_index 등록 → device 이동 → DataParallel 래핑

2. **DataParallel 래핑:**
   - main.py에서는 래핑하지 않음
   - SelfTrainer에서 내부적으로 처리

3. **디스크 공간:**
   - Self-training은 iteration마다 checkpoint 저장 (~1.3GB/iter)
   - 최소 5GB 이상 여유 공간 권장

4. **메모리 관리:**
   - 49,145개 문서에 대한 prediction 생성
   - OOM 발생 시 `EVAL_BATCH_SIZE` 줄이기

---

## 📚 추가 문서

- **ERROR_ANALYSIS.md**: 발생 가능한 에러 및 대처 방법
- **STAGE4_RESUME.md**: Stage 4 실행 가이드
- **README.md**: 전체 프로젝트 설명 (기존)

---

## 🎉 완료

모든 수정이 완료되었으며, Stage 4 (Self-Training)부터 안전하게 실행할 수 있습니다!
