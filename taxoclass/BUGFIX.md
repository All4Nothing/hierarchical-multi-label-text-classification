# Bug Fix: Multi-label Core Classes Support

## Issue

After implementing multi-label core class selection, the following error occurred:

```python
TypeError: unhashable type: 'list'
```

**Root Cause**: The `core_classes` dictionary structure changed from:
- **Old**: `Dict[int, int]` (doc_id → single class_id)
- **New**: `Dict[int, List[int]]` (doc_id → list of class_ids)

This caused type errors in code that expected single values instead of lists.

---

## Fixed Files

### 1. `main.py`

**Location**: Line 374-394 (wandb logging)

**Issue**: 
```python
unique_core_classes = len(set(core_classes.values()))  # ❌ TypeError: unhashable type: 'list'
```

**Fix**:
```python
# Flatten all core classes to get unique classes
all_core_classes = []
for doc_id, class_list in core_classes.items():
    all_core_classes.extend(class_list)

unique_core_classes = len(set(all_core_classes))  # ✅ Works with lists
total_core_class_assignments = len(all_core_classes)
avg_core_classes_per_doc = total_core_class_assignments / total_docs_with_core
```

**Updated Metrics**:
- `stage2/num_unique_core_classes`: Number of unique classes used as core classes
- `stage2/total_core_class_assignments`: Total core class assignments (sum across all docs)
- `stage2/avg_core_classes_per_doc`: Average core classes per document

---

### 2. `data/loader.py`

**Location**: `create_multi_label_matrix()` function

**Issue**: 
```python
def create_multi_label_matrix(
    doc_labels: List[int],
    core_class_assignments: Dict[int, int],  # ❌ Expected single int
    ...
```

Function iterated over single core_class values:
```python
for doc_id, core_class in core_class_assignments.items():
    # Processed single core_class
```

**Fix**:

1. **Updated signature** to accept multi-label format:
```python
def create_multi_label_matrix(
    doc_labels: List[int],
    core_class_assignments: Dict[int, List[int]],  # ✅ Now expects list
    ...
```

2. **Delegated to `create_training_labels()`** from `models.core_mining`:
```python
from models.core_mining import create_training_labels

label_matrix = create_training_labels(
    core_classes_dict=core_class_assignments,
    hierarchy=hierarchy,
    num_classes=num_classes,
    num_docs=num_docs  # Ensure correct size
)
```

**Benefits**:
- Uses the proper hierarchical label generation (positive/negative/ignore)
- Handles multiple core classes per document correctly
- Consistent with TaxoClass paper implementation

---

### 3. `models/core_mining.py`

**Location**: `create_training_labels()` function

**Issue**: 
```python
num_docs = len(core_classes_dict)  # ❌ Only covers docs with core classes
```

If `core_classes_dict` has 1000 entries but there are 2000 total docs, the label matrix would be (1000, num_classes) instead of (2000, num_classes).

**Fix**:

Added optional `num_docs` parameter:
```python
def create_training_labels(
    core_classes_dict: Dict[int, List[int]],
    hierarchy,
    num_classes: int,
    num_docs: int = None  # ✅ Now optional
) -> np.ndarray:
    """
    Args:
        num_docs: Total number of documents (if None, uses max(doc_id) + 1)
    """
    # Determine number of documents
    if num_docs is None:
        if core_classes_dict:
            num_docs = max(core_classes_dict.keys()) + 1
        else:
            num_docs = 0
    
    labels = np.zeros((num_docs, num_classes), dtype=np.float32)
    
    for doc_id in range(num_docs):
        if doc_id not in core_classes_dict:
            # Documents without core classes get all zeros (all negative)
            continue
        # Process core classes...
```

**Result**: Label matrix now always has correct shape (total_docs, num_classes).

---

## Testing

After fixes, the pipeline should run without errors:

```bash
python main.py
```

Expected output:
```
Stage 2: Core Class Mining
...
Identified core classes for 49145 documents
Total core classes: 208346, Avg per doc: 4.24

✅ Core classes saved: ./outputs/core_classes.npz

Stage 3: Classifier Training
Creating training labels from core classes...
100%|████████████████| 49145/49145 [00:02<00:00]

Label Statistics:
  Positive: 834123 (3.42%)
  Negative: 23456789 (96.12%)
  Ignore: 123456 (0.51%)
  Avg positive per doc: 16.98
```

---

## Type Compatibility

### Before (Single Core Class)
```python
core_classes = {0: 5, 1: 10, 2: 3}
core_class = core_classes[0]  # 5 (int)
```

### After (Multi-label Core Classes)
```python
core_classes = {0: [5, 6], 1: [10, 11, 12], 2: [3]}
core_classes_list = core_classes[0]  # [5, 6] (List[int])
```

### Migration Guide

If you have custom code using `core_classes`:

**Old Code**:
```python
# Get single core class
core = core_classes[doc_id]
if core != -1:
    print(f"Core class: {core}")
```

**New Code**:
```python
# Get list of core classes
cores = core_classes.get(doc_id, [])
if cores:
    print(f"Core classes: {cores}")
    for core in cores:
        # Process each core class
        pass
```

---

## Summary

✅ **Fixed**: TypeError when using multi-label core classes  
✅ **Fixed**: Label matrix size mismatch  
✅ **Fixed**: Wandb metrics for multi-label format  
✅ **Improved**: Consistent use of `create_training_labels()` function  

**Impact**: The pipeline now fully supports multi-label core class mining as intended by the TaxoClass paper.

---

## Verification

Run test to verify fixes:
```bash
cd /workspace/yongjoo/20252R0136DATA30400
python test_fixes.py
```

All tests should pass:
```
✅ Test 1: Multi-label core class selection - PASS
✅ Test 2: Hierarchical label generation - PASS  
✅ ALL TESTS PASSED
```
