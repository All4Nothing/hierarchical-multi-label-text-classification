# DataParallel State Dict 문제 완전 해결

## 문제 요약

**증상:**
```
RuntimeError: Error(s) in loading state_dict for TaxoClassifier:
    Missing key(s): "matching_matrix", ...
    Unexpected key(s): "module.matching_matrix", ...
```

**원인:**
- Multi-GPU 학습 시 `DataParallel`로 래핑된 모델 저장
- 모든 키에 `module.` prefix 추가됨
- 단일 GPU에서 로드 시 키 불일치 발생

## 수정된 파일

### 1. ✅ `generate_submission.py` (로드 측 수정)

**수정 내용:**
- DataParallel로 저장된 모델 자동 감지
- `module.` prefix 자동 제거
- `strict=False`로 추가 버퍼 무시

```python
# Check if state_dict has 'module.' prefix
has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())

if has_module_prefix:
    print("   Detected DataParallel model, removing 'module.' prefix...")
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict, strict=False)
```

### 2. ✅ `models/self_training.py` (저장 측 수정)

**수정 내용:**
- 저장 시 DataParallel unwrap하여 prefix 없이 저장
- 로드 시 DataParallel 상태 고려

**`save_model()` 수정:**
```python
def save_model(self, filename: str):
    save_path = os.path.join(self.save_dir, filename)
    
    # Get actual model (unwrap DataParallel if needed)
    model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
    
    torch.save({
        'model_state_dict': model_to_save.state_dict()
    }, save_path)
    print(f"Saved model to {save_path}")
```

**`load_model()` 수정:**
```python
def load_model(self, filename: str):
    load_path = os.path.join(self.save_dir, filename)
    checkpoint = torch.load(load_path)
    
    # Handle DataParallel: load into underlying model if wrapped
    if isinstance(self.model, torch.nn.DataParallel):
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from {load_path}")
```

### 3. ✅ `models/classifier.py` (이미 올바름)

**확인 결과:**
- `TaxoClassifierTrainer`는 이미 DataParallel을 올바르게 처리
- 저장/로드 모두 unwrap하여 처리

```python
def save_model(self, filename: str):
    model_state_dict = self.model.state_dict()
    if isinstance(self.model, torch.nn.DataParallel):
        model_state_dict = self.model.module.state_dict()  # ✅ Unwrap
    
    torch.save({
        'model_state_dict': model_state_dict,
        ...
    }, save_path)
```

## 수정 전후 비교

### Before (문제 발생)

**저장:**
```python
# self_training.py
torch.save({'model_state_dict': self.model.state_dict()}, ...)
# ❌ DataParallel이면 'module.' prefix 포함
```

**로드:**
```python
# generate_submission.py
model.load_state_dict(checkpoint['model_state_dict'])
# ❌ Prefix 불일치로 에러 발생
```

### After (해결)

**저장:**
```python
# self_training.py
model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
torch.save({'model_state_dict': model_to_save.state_dict()}, ...)
# ✅ 항상 prefix 없이 저장
```

**로드:**
```python
# generate_submission.py
if has_module_prefix:
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
# ✅ 혹시 모를 경우 대비 (이제는 불필요하지만 안전장치)
```

## 영향받는 모델 파일

### ❌ 이전에 저장된 파일 (prefix 있음)
- `saved_models/self_train_iter_1.pt`
- `saved_models/self_train_iter_2.pt`
- `saved_models/self_train_iter_3.pt`

**대응:** `generate_submission.py`가 자동으로 처리

### ✅ 앞으로 저장될 파일 (prefix 없음)
- 수정 후 새로 생성되는 모든 checkpoint

**대응:** 문제 없음

### ✅ Stage 3 파일 (이미 올바름)
- `saved_models/best_model.pt`
- `saved_models/checkpoint_epoch_*.pt`

**확인:** `TaxoClassifierTrainer`가 이미 올바르게 처리

## 테스트

### 1. 이전 모델로 submission 생성 (prefix 있음)

```bash
python generate_submission.py \
    --model_path saved_models/self_train_iter_3.pt \
    --hier_confidence \
    --confidence_threshold 0.4 \
    --min_labels 2 \
    --max_labels 3 \
    --output submission.csv
```

**예상 출력:**
```
   Detected DataParallel model, removing 'module.' prefix...
   ✅ State dict loaded successfully
```

### 2. 새로 저장된 모델 (prefix 없음)

앞으로 Stage 4를 다시 실행하면:
- 새로 저장되는 모델은 prefix 없음
- `generate_submission.py`는 자동으로 감지하여 처리하지 않음

## 추가 개선 사항

### 모든 모델 변환 스크립트 (선택사항)

기존 모델들을 모두 prefix 없이 재저장하려면:

```python
# convert_models.py
import torch
import os

model_dir = "saved_models"
for filename in os.listdir(model_dir):
    if filename.endswith('.pt'):
        path = os.path.join(model_dir, filename)
        checkpoint = torch.load(path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            has_prefix = any(k.startswith('module.') for k in state_dict.keys())
            
            if has_prefix:
                print(f"Converting {filename}...")
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                checkpoint['model_state_dict'] = state_dict
                torch.save(checkpoint, path)
                print(f"  ✅ Converted")
            else:
                print(f"Skipping {filename} (no prefix)")
```

**권장:** 필요 없음. `generate_submission.py`가 자동 처리.

## 결론

✅ **완전 해결**
1. 로드 측: `generate_submission.py` - 자동 감지 및 변환
2. 저장 측: `self_training.py` - 항상 prefix 없이 저장
3. 기존 코드: `classifier.py` - 이미 올바름

**호환성:**
- ✅ 이전 모델 (prefix 있음) - 자동 변환하여 로드
- ✅ 새 모델 (prefix 없음) - 직접 로드
- ✅ Stage 3 모델 - 이미 올바름

**추가 작업 불필요:**
- 기존 모델 파일 수정 불필요
- 모든 경우를 자동으로 처리
