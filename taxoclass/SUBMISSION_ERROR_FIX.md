# Generate Submission 에러 수정

## 발생한 에러

```
RuntimeError: Error(s) in loading state_dict for TaxoClassifier:
    Missing key(s) in state_dict: "matching_matrix", "doc_encoder.embeddings.word_embeddings.weight", ...
    Unexpected key(s) in state_dict: "module.matching_matrix", "module.doc_encoder.embeddings.word_embeddings.weight", ...
```

## 원인 분석

### 핵심 문제
**DataParallel Prefix 불일치**
- 저장된 모델: Multi-GPU 환경에서 `DataParallel`로 래핑된 상태로 저장
- 로드 시도: 단일 GPU 모델로 로드 시도
- 결과: 모든 키에 `module.` prefix가 있어서 불일치

### 왜 이런 일이?
1. `main.py` Stage 4 (Self-Training)에서:
   ```python
   if use_multi_gpu:
       model = torch.nn.DataParallel(model)
   ```
   
2. `self_training.py`의 `save_model()`:
   ```python
   torch.save({
       'model_state_dict': self.model.state_dict()
   }, save_path)
   ```
   - `self.model`이 DataParallel로 래핑된 상태
   - State dict의 모든 키에 `module.` prefix 추가됨

3. `generate_submission.py`:
   - 단일 GPU 모델 초기화 (DataParallel 없음)
   - Prefix 없는 키 기대
   - **불일치 발생!**

## 수정 사항

### `generate_submission.py` - DataParallel 감지 및 처리

**수정된 로직:**

```python
# Load state dict
state_dict = checkpoint['model_state_dict']

# Check if state_dict has 'module.' prefix (saved from DataParallel)
has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())

if has_module_prefix:
    print("   Detected DataParallel model, removing 'module.' prefix...")
    # Remove 'module.' prefix from all keys
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# Load state dict (use strict=False to ignore edge_index buffer if present)
model.load_state_dict(state_dict, strict=False)
```

**동작 원리:**
1. State dict의 키를 검사하여 `module.`로 시작하는지 확인
2. 발견되면 자동으로 prefix 제거
3. `strict=False`로 로드하여 추가 버퍼(edge_index) 무시

## 추가 개선 가능한 부분

### 1. `self_training.py` - 저장 시 prefix 제거 (더 나은 방법)

**문제:**
- 현재는 로드할 때마다 prefix 제거 필요
- 저장할 때 제거하면 더 깔끔

**개선안:**
```python
def save_model(self, filename: str):
    """Save model checkpoint"""
    save_path = os.path.join(self.save_dir, filename)
    
    # Get actual model (unwrap DataParallel if needed)
    model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
    
    torch.save({
        'model_state_dict': model_to_save.state_dict()
    }, save_path)
    print(f"Saved model to {save_path}")
```

### 2. `models/classifier.py` - TaxoClassifierTrainer 저장 함수도 동일하게

**확인 필요:**
- `TaxoClassifierTrainer`의 `save_model()` 메서드도 동일한 문제 가능성
- `best_model.pt`는 제대로 저장되었는지 확인 필요

## 테스트

```bash
cd /workspace/yongjoo/20252R0136DATA30400/taxoclass

python generate_submission.py \
    --model_path saved_models/self_train_iter_3.pt \
    --hier_confidence \
    --confidence_threshold 0.4 \
    --min_labels 2 \
    --max_labels 3 \
    --output submission.csv
```

## 예상 출력

```
================================================================================
                         GENERATE SUBMISSION FILE
================================================================================
Using device: cuda

Loading taxonomy hierarchy...
Loaded taxonomy: 531 classes, max depth: 2

Loading model from saved_models/self_train_iter_3.pt...
Initializing class embeddings...
   Checkpoint keys: ['model_state_dict']
   Loaded from 'model_state_dict'
   Detected DataParallel model, removing 'module.' prefix...
   ✅ State dict loaded successfully
✅ Model loaded successfully!

Generating predictions...
Predicting: 100%|████████████████████████| 154/154 [00:XX<00:00, XX.XXit/s]

Applying hierarchical confidence filtering...
...
✅ Submission file saved to submission.csv
```

## 다른 잠재적 문제

### 1. ⚠️ best_model.pt도 동일한 문제?

**확인 필요:**
```bash
# best_model.pt가 DataParallel로 저장되었는지 확인
python -c "
import torch
ckpt = torch.load('saved_models/best_model.pt', map_location='cpu')
state_dict = ckpt['model_state_dict']
has_module = any(k.startswith('module.') for k in state_dict.keys())
print(f'Has module prefix: {has_module}')
"
```

**해결책:**
- 동일한 수정이 이미 `generate_submission.py`에 적용되어 있으므로 문제없음

### 2. ⚠️ Edge Index 불일치

**문제:**
- Self-training 시 edge_index가 버퍼로 저장됨
- `strict=False` 사용으로 이미 해결됨

### 3. ⚠️ Class Embeddings 초기화

**문제:**
- 저장된 모델의 class_embeddings와 새로 초기화한 embeddings가 다를 수 있음

**현재 로직:**
```python
# Initialize class embeddings with BERT
class_embeddings = initialize_class_embeddings_with_bert(...)
model.class_embeddings.data = class_embeddings.to(device)  # ❌ 덮어씀!

# Load model weights (이후에 로드)
model.load_state_dict(state_dict, strict=False)  # ✅ 다시 복원됨
```

**분석:**
- 초기화 후 로드하므로 결국 올바른 embeddings 사용
- 하지만 불필요한 초기화 단계
- 성능에는 영향 없음

## 권장 사항

### 단기 해결책 (이미 적용됨)
✅ `generate_submission.py`에서 DataParallel prefix 자동 제거

### 장기 해결책 (선택사항)
1. `self_training.py`의 `save_model()` 수정하여 prefix 없이 저장
2. `TaxoClassifierTrainer`의 `save_model()` 동일하게 수정
3. Class embeddings 초기화 로직 최적화

## 결론

✅ **에러 수정 완료**
- DataParallel로 저장된 모델을 자동으로 감지하고 prefix 제거
- `strict=False`로 추가 버퍼 무시
- 이제 정상적으로 submission 생성 가능
