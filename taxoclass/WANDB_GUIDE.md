# Weights & Biases (wandb) Integration Guide

## 🎯 **Wandb를 통한 학습 모니터링**

이 프로젝트는 Weights & Biases (wandb)를 통해 전체 학습 과정을 실시간으로 모니터링할 수 있도록 구성되었습니다.

---

## 📦 **설치**

### **1. wandb 설치**

```bash
pip install wandb
```

또는 requirements.txt를 통해 설치:

```bash
pip install -r requirements.txt
```

### **2. wandb 로그인**

```bash
wandb login
```

- 브라우저가 열리면 wandb 계정으로 로그인
- API 키를 복사하여 터미널에 붙여넣기
- 또는 [https://wandb.ai/authorize](https://wandb.ai/authorize)에서 직접 API 키 확인

---

## 🚀 **사용 방법**

### **Option 1: config.py에서 활성화 (권장)**

```python
# config.py
USE_WANDB = True  # wandb 사용 활성화
WANDB_PROJECT = "taxoclass-hierarchical"  # 프로젝트 이름
WANDB_ENTITY = None  # 팀 이름 (개인 계정은 None)
WANDB_RUN_NAME = None  # Run 이름 (자동 생성됨)
WANDB_TAGS = ["hierarchical", "taxonomy", "gnn"]  # 태그
```

```bash
# 그대로 실행
python main.py
```

### **Option 2: 코드 실행 시 제어**

```python
# config.py에서 USE_WANDB = False로 설정
USE_WANDB = False

# 실행 시 환경변수로 제어
export USE_WANDB=1  # 또는
WANDB_MODE=offline python main.py  # 오프라인 모드
```

### **Option 3: wandb 없이 실행**

```bash
# wandb를 설치하지 않거나 USE_WANDB=False
python main.py
# → 자동으로 wandb 없이 실행됨 (경고 메시지만 출력)
```

---

## 📊 **로깅되는 메트릭**

### **1. Stage 1: Similarity Calculation**

```python
stage1/similarity_min      # Similarity 최소값
stage1/similarity_max      # Similarity 최대값
stage1/similarity_mean     # Similarity 평균
stage1/similarity_std      # Similarity 표준편차
```

**분석**:
- Similarity 분포를 통해 zero-shot classification 품질 파악
- Mean이 너무 낮으면 모델과 데이터 간 mismatch
- Std가 너무 작으면 모든 클래스가 비슷하게 보임

---

### **2. Stage 2: Core Class Mining**

```python
stage2/num_core_classes           # Core class 개수
stage2/total_docs_with_core       # Core가 할당된 문서 수
stage2/avg_docs_per_core_class    # Core class당 평균 문서 수
```

**분석**:
- Core class 개수가 너무 많으면: Threshold 너무 낮음
- Core class 개수가 너무 적으면: Threshold 너무 높음
- 이상적: 전체 class의 20-30%가 core class

---

### **3. Stage 3: Classifier Training**

#### **데이터 통계**
```python
stage3/num_positive_labels    # Positive label 개수
stage3/num_negative_labels    # Negative label 개수
stage3/num_ignored_labels     # Ignored label (-1) 개수
stage3/positive_ratio         # Positive label 비율
stage3/train_samples          # Training sample 수
stage3/val_samples            # Validation sample 수
```

#### **학습 과정 (실시간)**
```python
stage3/train_loss             # Training loss (매 10 step)
stage3/learning_rate          # Learning rate (매 10 step)
stage3/epoch                  # 현재 epoch
```

#### **Epoch별 메트릭**
```python
stage3/epoch_train_loss       # Epoch 평균 train loss
stage3/epoch_val_loss         # Epoch 평균 validation loss
stage3/best_val_loss          # 최고 validation loss
stage3/best_epoch             # 최고 성능 epoch
```

**분석**:
- Train loss vs Val loss: Overfitting 여부 확인
- Learning rate schedule: Warmup 후 감소 확인
- Best epoch: Early stopping이 적절한지 확인

---

### **4. Stage 4: Self-Training**

#### **Iteration별 통계**
```python
stage4/confident_predictions  # Confident sample 수
stage4/confidence_ratio       # Confidence 비율 (%)
stage4/avg_max_prediction     # 평균 최대 예측값
stage4/avg_target_entropy     # 평균 target entropy
stage4/iteration              # 현재 iteration
```

#### **학습 과정**
```python
stage4/self_train_loss        # Self-training loss
stage4/epoch                  # Epoch within iteration
```

**분석**:
- Confident predictions 증가 추세: 모델이 확신 증가
- Avg max prediction 증가: Pseudo-label 품질 향상
- Avg target entropy 감소: 더 sharp한 예측

---

### **5. Test (Final Evaluation)**

```python
test/accuracy                 # Test accuracy
test/precision                # Precision
test/recall                   # Recall
test/f1_score                 # F1 score
test/top3_accuracy            # Top-3 accuracy
test/top5_accuracy            # Top-5 accuracy
test/top10_accuracy           # Top-10 accuracy
test/level_0_accuracy         # Level 0 accuracy
test/level_1_accuracy         # Level 1 accuracy
...
```

**분석**:
- Overall accuracy: 최종 성능
- Top-k accuracy: Ranking 품질
- Level-wise accuracy: 계층별 성능

---

## 📈 **Wandb Dashboard 활용**

### **1. 실시간 모니터링**

```
Run Page → Charts 탭
- Train loss curve
- Validation loss curve
- Learning rate schedule
- Confidence ratio over iterations
```

**유용한 차트**:

#### **Loss Curves (Stage 3)**
```python
# X-axis: global_step
# Y-axis: stage3/train_loss, stage3/epoch_val_loss
# → Overfitting 감지
```

#### **Self-Training Progress (Stage 4)**
```python
# X-axis: iteration
# Y-axis: stage4/confidence_ratio, stage4/avg_max_prediction
# → Pseudo-label 품질 개선 확인
```

---

### **2. 실험 비교**

```
Project Page → Runs 탭
- 여러 run을 선택하여 비교
- Parallel coordinates plot
- Scatter plot matrix
```

**비교 예시**:

```python
# 실험 1: bert-base + GNN 3-layer
# 실험 2: bert-large + GNN 4-layer
# 실험 3: bert-large + GNN 4-layer + Test data in Stage 3

# 비교 메트릭:
# - test/accuracy
# - stage3/best_val_loss
# - stage4/confidence_ratio
# - Training time
```

---

### **3. Hyperparameter Tuning**

```
Sweeps 탭 → Create sweep
```

**권장 sweep 설정**:

```yaml
# sweep.yaml
program: main.py
method: bayes  # or grid, random
metric:
  name: test/accuracy
  goal: maximize
parameters:
  learning_rate:
    min: 5e-6
    max: 5e-5
  gnn_hidden_dim:
    values: [512, 768, 1024]
  gnn_num_layers:
    values: [3, 4, 5]
  self_train_threshold:
    min: 0.4
    max: 0.7
```

```bash
# Sweep 시작
wandb sweep sweep.yaml
wandb agent <sweep_id>
```

---

## 🎨 **Custom Visualizations**

### **1. Confusion Matrix (추가 가능)**

```python
# main.py에 추가
if use_wandb:
    # Compute confusion matrix
    wandb.log({
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=true_labels,
            preds=pred_labels,
            class_names=class_names
        )
    })
```

### **2. Class-wise Performance (추가 가능)**

```python
# main.py에 추가
if use_wandb:
    # Create table
    table = wandb.Table(
        columns=["Class", "Precision", "Recall", "F1", "Support"],
        data=class_metrics
    )
    wandb.log({"test/class_performance": table})
```

### **3. Prediction Examples (추가 가능)**

```python
# main.py에 추가
if use_wandb:
    # Log sample predictions
    examples = []
    for i in range(10):
        examples.append([
            test_documents[i],
            true_labels[i],
            pred_labels[i],
            "✓" if true_labels[i] == pred_labels[i] else "✗"
        ])
    
    table = wandb.Table(
        columns=["Document", "True", "Predicted", "Correct"],
        data=examples
    )
    wandb.log({"test/prediction_examples": table})
```

---

## 🔧 **고급 설정**

### **1. Offline Mode (네트워크 없이)**

```bash
# 오프라인 모드로 실행
export WANDB_MODE=offline
python main.py

# 나중에 sync
wandb sync wandb/latest-run
```

### **2. Custom Run Name**

```python
# config.py
WANDB_RUN_NAME = "bert-large_gnn4_a6000_v1"
```

또는

```python
# main.py에서 동적 생성
run_name = f"taxo_{Config.DOC_ENCODER_MODEL.split('/')[-1]}_gnn{Config.GNN_NUM_LAYERS}_lr{Config.LEARNING_RATE}"
```

### **3. Group & Tags**

```python
# config.py
WANDB_TAGS = ["bert-large", "a6000", "transductive", "gnn"]

# main.py
wandb.init(
    project=Config.WANDB_PROJECT,
    name=run_name,
    tags=Config.WANDB_TAGS,
    group="bert-large_experiments",  # 그룹으로 묶기
)
```

### **4. Resume Training**

```python
# main.py
wandb.init(
    project=Config.WANDB_PROJECT,
    id="unique_run_id",  # 이전 run의 ID
    resume="must"  # 반드시 resume
)
```

---

## 🐛 **Troubleshooting**

### **문제 1: wandb가 설치되지 않음**

```bash
# 증상
ModuleNotFoundError: No module named 'wandb'

# 해결
pip install wandb
```

### **문제 2: 로그인 안됨**

```bash
# 증상
wandb: ERROR Unable to authenticate

# 해결
wandb login
# 또는
export WANDB_API_KEY=<your_api_key>
```

### **문제 3: 네트워크 연결 실패**

```bash
# 증상
wandb: WARNING Network error

# 해결 (오프라인 모드)
export WANDB_MODE=offline
python main.py
```

### **문제 4: 로그가 너무 많음**

```python
# config.py
WANDB_LOG_INTERVAL = 100  # 10 → 100 (덜 자주 로깅)
WANDB_LOG_GRADIENTS = False  # Gradient 로깅 끄기
```

### **문제 5: wandb 완전히 비활성화**

```python
# config.py
USE_WANDB = False

# 또는 환경변수
export WANDB_MODE=disabled
python main.py
```

---

## 📊 **예제: 실험 결과 분석**

### **Scenario: bert-base vs bert-large 비교**

#### **Run 1: bert-base**
```python
# config.py
DOC_ENCODER_MODEL = "bert-base-uncased"
EMBEDDING_DIM = 768
GNN_HIDDEN_DIM = 512
BATCH_SIZE = 32

# 결과 (wandb)
test/accuracy: 0.752
stage3/best_val_loss: 0.348
Training time: 3.2 hours
```

#### **Run 2: bert-large (A6000 최적화)**
```python
# config.py
DOC_ENCODER_MODEL = "bert-large-uncased"
EMBEDDING_DIM = 1024
GNN_HIDDEN_DIM = 1024
BATCH_SIZE = 64

# 결과 (wandb)
test/accuracy: 0.817  # +6.5%p 향상! ⭐
stage3/best_val_loss: 0.291  # 더 낮은 loss
Training time: 2.1 hours  # Mixed precision 덕분에 더 빠름!
```

#### **Wandb 비교 차트**

```
Compare Runs:
- X-axis: training time
- Y-axis: test/accuracy
- Color: model (bert-base vs bert-large)

→ bert-large가 더 빠르고 성능도 높음!
```

---

## 🎯 **Best Practices**

### **1. 체계적인 실험 관리**

```python
# 명확한 run name
run_name = f"{model_name}_gnn{n_layers}_lr{lr}_bs{batch_size}_v{version}"

# 유의미한 tags
tags = ["baseline", "bert-large", "a6000", "transductive"]

# 실험 그룹
group = "bert-large_ablation"  # 같은 실험군
```

### **2. 중요 메트릭 우선**

```python
# Summary에 최종 결과 기록
wandb.run.summary["final_accuracy"] = test_accuracy
wandb.run.summary["final_f1"] = test_f1
wandb.run.summary["training_time_hours"] = training_time
```

### **3. 재현 가능성 확보**

```python
# Config 저장
wandb.config.update({
    "seed": Config.SEED,
    "git_commit": get_git_commit(),  # Git commit hash
    "cuda_version": torch.version.cuda,
    "pytorch_version": torch.__version__,
})

# Artifacts로 모델 저장
artifact = wandb.Artifact("taxo_model", type="model")
artifact.add_file("saved_models/best_model.pt")
wandb.log_artifact(artifact)
```

---

## 📚 **추가 자료**

- **Wandb 공식 문서**: [https://docs.wandb.ai](https://docs.wandb.ai)
- **PyTorch Integration**: [https://docs.wandb.ai/guides/integrations/pytorch](https://docs.wandb.ai/guides/integrations/pytorch)
- **Sweeps Guide**: [https://docs.wandb.ai/guides/sweeps](https://docs.wandb.ai/guides/sweeps)

---

## ✅ **Quick Start Checklist**

- [ ] wandb 설치: `pip install wandb`
- [ ] 로그인: `wandb login`
- [ ] config.py에서 `USE_WANDB = True` 설정
- [ ] 프로젝트 이름 설정: `WANDB_PROJECT = "your-project"`
- [ ] 학습 실행: `python main.py`
- [ ] Dashboard 확인: [https://wandb.ai](https://wandb.ai)
- [ ] 실험 비교 및 분석

---

**마지막 업데이트**: 2025-11-22  
**버전**: 1.0  
**Wandb를 통해 효율적인 실험 관리를 경험하세요!** 🚀

