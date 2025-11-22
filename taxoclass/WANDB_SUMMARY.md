# Wandb 통합 요약

## ✅ **완료된 작업**

### **1. Config 설정 추가** (`config.py`)

```python
# Weights & Biases (wandb) Settings
USE_WANDB = True
WANDB_PROJECT = "taxoclass-hierarchical"
WANDB_ENTITY = None
WANDB_RUN_NAME = None
WANDB_TAGS = ["hierarchical", "taxonomy", "gnn"]
WANDB_LOG_INTERVAL = 10
WANDB_LOG_GRADIENTS = False
```

---

### **2. Main Pipeline 수정** (`main.py`)

#### **Wandb 초기화**
- Auto-generated run name: `taxo_bert-large_gnn4_h1024`
- Full config logging (모든 hyperparameters)
- Tags 및 metadata 추가

#### **Stage별 로깅**
- **Stage 1**: Similarity 통계 (min, max, mean, std)
- **Stage 2**: Core class 통계 (num, total docs, avg docs)
- **Stage 3**: Label 분포, Train/Val samples
- **Final**: Test metrics (accuracy, F1, Top-k)

---

### **3. Classifier Trainer 수정** (`models/classifier.py`)

#### **실시간 로깅**
- Training loss (매 10 step)
- Learning rate schedule
- Epoch별 train/val loss
- Best model update 기록

#### **Global step tracking**
- Continuous step counter
- Epoch 구분 가능

---

### **4. Self-Trainer 수정** (`models/self_training.py`)

#### **Iteration별 로깅**
- Confident predictions 수 및 비율
- Avg max prediction (confidence)
- Avg target entropy
- Self-training loss per epoch

---

### **5. Requirements 업데이트** (`requirements.txt`)

```
wandb>=0.15.0
```

---

### **6. 가이드 문서 작성** (`WANDB_GUIDE.md`)

- 설치 및 설정 방법
- 로깅되는 메트릭 상세 설명
- Dashboard 활용법
- Troubleshooting
- Best practices

---

## 📊 **로깅 구조**

```
taxoclass-hierarchical (Project)
│
├── stage1/
│   ├── similarity_min
│   ├── similarity_max
│   ├── similarity_mean
│   └── similarity_std
│
├── stage2/
│   ├── num_core_classes
│   ├── total_docs_with_core
│   └── avg_docs_per_core_class
│
├── stage3/
│   ├── num_positive_labels
│   ├── num_negative_labels
│   ├── train_samples
│   ├── val_samples
│   ├── train_loss (real-time)
│   ├── learning_rate (real-time)
│   ├── epoch_train_loss
│   ├── epoch_val_loss
│   ├── best_val_loss
│   └── best_epoch
│
├── stage4/
│   ├── confident_predictions
│   ├── confidence_ratio
│   ├── avg_max_prediction
│   ├── avg_target_entropy
│   ├── self_train_loss
│   └── iteration
│
└── test/
    ├── accuracy
    ├── precision
    ├── recall
    ├── f1_score
    ├── top3_accuracy
    ├── top5_accuracy
    ├── top10_accuracy
    └── level_*_accuracy
```

---

## 🚀 **사용 방법**

### **Quick Start**

```bash
# 1. wandb 설치
pip install wandb

# 2. 로그인
wandb login

# 3. 학습 실행 (config.py에서 USE_WANDB=True)
python main.py

# 4. Dashboard 확인
# → 터미널에 출력된 URL 클릭
```

### **Wandb 없이 실행**

```python
# config.py
USE_WANDB = False
```

또는

```bash
export WANDB_MODE=disabled
python main.py
```

---

## 💡 **주요 기능**

### **1. 실시간 모니터링**
- Training loss curve
- Validation loss curve
- Learning rate schedule
- Self-training confidence progression

### **2. 실험 비교**
- 여러 run을 한 번에 비교
- Hyperparameter sweep 지원
- Best model tracking

### **3. 재현 가능성**
- 모든 config 자동 저장
- Git commit hash 기록 (추가 가능)
- Random seed 저장

---

## 📈 **예상 효과**

### **실험 관리**
- ✅ 여러 실험을 체계적으로 관리
- ✅ Best model 자동 추적
- ✅ Hyperparameter 영향 분석

### **디버깅**
- ✅ Loss curve로 overfitting 감지
- ✅ Learning rate schedule 확인
- ✅ Confidence progression 모니터링

### **성능 최적화**
- ✅ Hyperparameter tuning 효율화
- ✅ 실험 결과 쉬운 비교
- ✅ Ablation study 용이

---

## 🔧 **확장 가능성**

### **추가 가능한 로깅**

#### **1. Gradient Histograms**
```python
# config.py
WANDB_LOG_GRADIENTS = True

# models/classifier.py
if self.use_wandb and Config.WANDB_LOG_GRADIENTS:
    wandb.watch(self.model, log="all", log_freq=100)
```

#### **2. Model Artifacts**
```python
# main.py
if use_wandb:
    artifact = wandb.Artifact("best_model", type="model")
    artifact.add_file("saved_models/best_model.pt")
    wandb.log_artifact(artifact)
```

#### **3. Prediction Examples**
```python
# main.py
if use_wandb:
    table = wandb.Table(
        columns=["Document", "True", "Predicted", "Correct"],
        data=prediction_examples
    )
    wandb.log({"test/predictions": table})
```

#### **4. Confusion Matrix**
```python
# main.py
if use_wandb:
    wandb.log({
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=true_labels,
            preds=pred_labels,
            class_names=class_names
        )
    })
```

---

## 📝 **수정된 파일 목록**

1. ✅ `config.py` - Wandb 설정 추가
2. ✅ `main.py` - Wandb 초기화 및 로깅
3. ✅ `models/classifier.py` - Trainer 로깅
4. ✅ `models/self_training.py` - Self-trainer 로깅
5. ✅ `requirements.txt` - wandb 추가
6. ✅ `WANDB_GUIDE.md` - 상세 가이드 (36페이지)
7. ✅ `WANDB_SUMMARY.md` - 요약 문서 (현재 파일)

---

## 🎯 **테스트 체크리스트**

- [ ] wandb 설치 확인: `pip list | grep wandb`
- [ ] 로그인 확인: `wandb login`
- [ ] 학습 실행: `python main.py`
- [ ] Wandb run 생성 확인
- [ ] Dashboard URL 접속
- [ ] Stage별 메트릭 확인
- [ ] Loss curve 확인
- [ ] Final test metrics 확인

---

**구현 완료!** 🎉  
이제 wandb를 통해 전체 학습 과정을 실시간으로 모니터링하고, 여러 실험을 체계적으로 관리할 수 있습니다.

