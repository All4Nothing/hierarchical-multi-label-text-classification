# TELEClass Pipeline Flow Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                                   │
├─────────────────────────────────────────────────────────────────────┤
│  • train_corpus.txt       (29,487 documents)                        │
│  • test_corpus.txt        (19,658 documents)                        │
│  • class_hierarchy.txt    (568 edges)                               │
│  • class_related_keywords.txt (531 classes)                         │
│  • classes.txt            (531 class names)                         │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 0: REPRODUCIBILITY                          │
├─────────────────────────────────────────────────────────────────────┤
│  set_seed(42)                                                        │
│  • Python random                                                     │
│  • NumPy                                                             │
│  • PyTorch (CPU + CUDA)                                              │
│  • CuDNN (deterministic mode)                                        │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA LOADING                                     │
├─────────────────────────────────────────────────────────────────────┤
│  DataLoader.load_all()                                               │
│  • Parse all 5 input files                                           │
│  • Combine: all_corpus = train_corpus + test_corpus                 │
│  • Build NetworkX hierarchy graph                                    │
│  • Create class-keyword mappings                                     │
│  • Create class ↔ index mappings                                     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│           PHASE 1: CLASS REPRESENTATION (Contextual Injection)       │
├─────────────────────────────────────────────────────────────────────┤
│  Model: sentence-transformers/all-mpnet-base-v2                     │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ For each class:                                          │        │
│  │   description = "The product category is {class},       │        │
│  │                  associated with keywords: {keywords}"   │        │
│  └─────────────────────────────────────────────────────────┘        │
│                           │                                           │
│                           ▼                                           │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ Encode class descriptions                                │        │
│  │   → class_embeddings [531 × 768]                        │        │
│  └─────────────────────────────────────────────────────────┘        │
│                           │                                           │
│                           ▼                                           │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ Encode ALL documents (train + test)                     │        │
│  │   → doc_embeddings [49,145 × 768]                       │        │
│  │                                                           │        │
│  │ ★ TRANSDUCTIVE LEARNING: Test data included here!       │        │
│  └─────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│        PHASE 2: ITERATIVE PSEUDO-LABELING (Refinement Loop)         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Initial: Compute cosine similarity                                  │
│    similarity = doc_embeddings @ class_embeddings.T                 │
│    → [49,145 × 531]                                                  │
│                                                                       │
│  ┌───────────────────────────────────────────────────────┐          │
│  │  FOR iteration = 1 to 3:                              │          │
│  │                                                         │          │
│  │    Step A: Select Reliable Documents                   │          │
│  │    ────────────────────────────────                    │          │
│  │    For each class c:                                   │          │
│  │      • Get top-20 most confident documents             │          │
│  │      • These are "reliable" pseudo-labeled samples     │          │
│  │                                                         │          │
│  │    Step B: Update Class Centroids                      │          │
│  │    ─────────────────────────────                       │          │
│  │    For each class c:                                   │          │
│  │      • Compute mean of reliable doc embeddings         │          │
│  │      • Update: class_embeddings[c] = mean(reliable)    │          │
│  │                                                         │          │
│  │    ★ This aligns classes with actual data distribution!│          │
│  │    ★ Uses BOTH train and test docs for alignment!      │          │
│  │                                                         │          │
│  │    Recompute similarity with updated centroids         │          │
│  │                                                         │          │
│  └───────────────────────────────────────────────────────┘          │
│                           │                                           │
│                           ▼                                           │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ Final Pseudo-Labeling (Gap-Based)                       │        │
│  │ ─────────────────────────────────────                   │        │
│  │ For each document:                                       │        │
│  │   1. Sort classes by similarity (descending)            │        │
│  │   2. Compute gaps: diff[i] = score[i] - score[i+1]     │        │
│  │   3. Find largest gap in top-5                          │        │
│  │   4. Assign labels up to gap (min 2 labels)             │        │
│  │                                                           │        │
│  │ Example:                                                 │        │
│  │   Scores: [0.85, 0.82, 0.78, 0.45, 0.32, ...]          │        │
│  │   Gaps:   [0.03, 0.04, 0.33*, 0.13, ...]               │        │
│  │   Cutoff: 3 labels (gap at position 2)                  │        │
│  └─────────────────────────────────────────────────────────┘        │
│                           │                                           │
│                           ▼                                           │
│  Output: pseudo_labels for ALL 49,145 documents                      │
│          pseudo_scores (confidence values)                           │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│            PHASE 3: DATA AUGMENTATION (Starved Classes)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Step 1: Identify starved classes                                    │
│    • Count documents per class (train only)                          │
│    • Starved = classes with < 10 documents                           │
│                                                                       │
│  Step 2: Generate synthetic data (Placeholder)                       │
│    • Get taxonomy path for each starved class                        │
│    • [Optional] Call LLM API to generate reviews                     │
│    • [Future] Add synthetic docs to training set                     │
│                                                                       │
│  Current: Returns empty (augmentation optional)                      │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              PHASE 4: HIERARCHY EXPANSION                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  For each document's pseudo-labels:                                  │
│    For each label L:                                                 │
│      • Use NetworkX to find all ancestors                            │
│      • Add ancestors to label set                                    │
│                                                                       │
│  Example:                                                            │
│    Input:  ["Baby Cereal"]                                           │
│    Hierarchy: 0 → 1 → 2  (Baby Product → Feeding → Baby Cereal)    │
│    Output: ["Baby Cereal", "Feeding", "Baby Product"]               │
│                                                                       │
│  Effect: Average labels per doc increases (e.g., 3.2 → 5.7)         │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              PHASE 5: BERT CLASSIFIER TRAINING                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Training Data:                                                      │
│    • train_corpus (29,487 docs)                                      │
│    • Expanded pseudo-labels from Phase 4                             │
│    • [Optional] Augmented data from Phase 3                          │
│                                                                       │
│  Model: bert-base-uncased                                            │
│    • Classification head: 531 outputs (one per class)                │
│    • Loss: BCEWithLogitsLoss (multi-label)                           │
│    • Optimizer: AdamW (lr=2e-5)                                      │
│    • Scheduler: Linear warmup (10% steps)                            │
│                                                                       │
│  Training Loop (3 epochs):                                           │
│    ┌─────────────────────────────────────────────┐                  │
│    │  FOR epoch = 1 to 3:                        │                  │
│    │    FOR batch in train_loader:               │                  │
│    │      1. Forward pass                         │                  │
│    │      2. Compute BCEWithLogitsLoss            │                  │
│    │      3. Backward pass                        │                  │
│    │      4. Gradient clipping                    │                  │
│    │      5. Optimizer step                       │                  │
│    │      6. Scheduler step                       │                  │
│    │                                               │                  │
│    │    [Optional] Validation                     │                  │
│    │    Save best model                           │                  │
│    └─────────────────────────────────────────────┘                  │
│                                                                       │
│  Output: Trained BERT model saved to outputs/models/best_model/     │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│            PHASE 6: INFERENCE & SUBMISSION GENERATION                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Input: test_corpus (19,658 documents)                               │
│                                                                       │
│  Step 1: Load trained BERT model                                     │
│    • Load from outputs/models/best_model/                            │
│                                                                       │
│  Step 2: Batch inference                                             │
│    ┌─────────────────────────────────────────────┐                  │
│    │  FOR batch in test_corpus:                  │                  │
│    │    1. Tokenize texts                        │                  │
│    │    2. Forward pass through BERT             │                  │
│    │    3. Apply sigmoid to logits               │                  │
│    │    4. Threshold at 0.5                      │                  │
│    │    5. Get predicted label indices           │                  │
│    └─────────────────────────────────────────────┘                  │
│                                                                       │
│  Step 3: Hierarchy expansion                                         │
│    • Apply same expansion as Phase 4                                 │
│    • Add ancestor classes to predictions                             │
│                                                                       │
│  Step 4: Generate Kaggle submission CSV                              │
│    ┌──────────────────────────────┐                                 │
│    │ ID,Label                     │                                 │
│    │ 0,class1 class2 class3       │                                 │
│    │ 1,class4 class5              │                                 │
│    │ ...                          │                                 │
│    └──────────────────────────────┘                                 │
│                                                                       │
│  Output: outputs/submission.csv                                      │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FINAL OUTPUT                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ✓ submission.csv         - Kaggle submission file                  │
│  ✓ best_model/            - Trained BERT checkpoint                 │
│  ✓ intermediate results   - Embeddings, pseudo-labels               │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Visualization

```
┌──────────────┐     ┌──────────────┐
│ Train Corpus │     │ Test Corpus  │
│  (29,487)    │     │  (19,658)    │
└──────┬───────┘     └──────┬───────┘
       │                    │
       └────────┬───────────┘
                │
                ▼
        ┌──────────────┐
        │ All Corpus   │
        │  (49,145)    │◄─────── ★ TRANSDUCTIVE LEARNING
        └──────┬───────┘         (Test data used for
               │                  representation learning)
               ▼
        ┌──────────────┐
        │  Embeddings  │
        │ [49145×768]  │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ Refinement   │───► Pseudo-labels for ALL docs
        │   Loop       │     (train + test)
        └──────┬───────┘
               │
               ├──────────────┐
               │              │
               ▼              ▼
    ┌────────────────┐  ┌────────────────┐
    │ Train Labels   │  │ Test Labels    │
    │   (29,487)     │  │   (19,658)     │
    └────────┬───────┘  └────────────────┘
             │                    │
             ▼                    │
    ┌────────────────┐            │
    │ BERT Training  │            │
    │  (3 epochs)    │            │
    └────────┬───────┘            │
             │                    │
             ▼                    │
    ┌────────────────┐            │
    │ Trained Model  │            │
    └────────┬───────┘            │
             │                    │
             └──────────┬─────────┘
                        │
                        ▼
                ┌──────────────┐
                │  Inference   │
                │  on Test     │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │ Submission   │
                │    CSV       │
                └──────────────┘
```

## Key Components Interaction

```
┌───────────────────────────────────────────────────────────────────┐
│                       DataLoader                                   │
│  • Parses 5 input files                                            │
│  • Combines train + test                                           │
│  • Builds hierarchy graph                                          │
└───────────┬───────────────────────────────────────────────────────┘
            │
            ├──────────────────────────────────────────────────────┐
            │                                                       │
            ▼                                                       ▼
┌────────────────────────┐                        ┌────────────────────────┐
│ ClassRepresentation    │                        │ HierarchyExpander      │
│ Module                 │                        │                        │
│  • Encodes classes     │                        │  • NetworkX BFS        │
│  • Encodes documents   │                        │  • Adds ancestors      │
└───────────┬────────────┘                        └────────┬───────────────┘
            │                                              │
            ▼                                              │
┌────────────────────────┐                                │
│ IterativePseudo        │                                │
│ Labeler                │                                │
│  • Similarity matrix   │                                │
│  • Refinement loop     │────────────────────────────────┤
│  • Gap-based cutoff    │                                │
└───────────┬────────────┘                                │
            │                                              │
            ├──────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────┐
│ BERTClassifier         │
│ Trainer                │
│  • Multi-label BERT    │
│  • BCEWithLogitsLoss   │
│  • AdamW + Warmup      │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ InferenceModule        │
│  • Load trained model  │
│  • Batch prediction    │
│  • CSV generation      │
└────────────────────────┘
```

## Timeline & Checkpoints

```
Start
  │
  ├─ [0-5 min]   Phase 1: Encoding
  │                ├─ Encode 531 classes
  │                └─ Encode 49,145 documents
  │
  ├─ [5-10 min]  Phase 2: Refinement
  │                ├─ Iteration 1
  │                ├─ Iteration 2
  │                ├─ Iteration 3
  │                └─ Final pseudo-labeling
  │
  ├─ [10-11 min] Phase 3: Augmentation
  │                └─ Identify starved classes
  │
  ├─ [11-12 min] Phase 4: Hierarchy Expansion
  │                └─ Expand all pseudo-labels
  │
  ├─ [12-72 min] Phase 5: BERT Training
  │                ├─ Epoch 1 (20 min)
  │                ├─ Epoch 2 (20 min)
  │                └─ Epoch 3 (20 min)
  │
  ├─ [72-82 min] Phase 6: Inference
  │                ├─ Predict on 19,658 test docs
  │                ├─ Expand predictions
  │                └─ Generate submission.csv
  │
End (Total: ~80 min on GPU)
```

## Critical Design Patterns

### 1. Transductive Learning Pattern

```
Traditional Approach:
  train_data → model → test_data
      ↓                    ↑
  supervised          inference only

Our Approach (Transductive):
  train_data + test_data → unsupervised → pseudo-labels
                ↓                              ↓
         representation                   supervised
           learning                        fine-tuning
                ↓                              ↓
            model ← ← ← ← ← ← ← ← ← ← ← ← test_pred
```

### 2. Iterative Refinement Pattern

```
Initial:  Class descriptions → Static embeddings
              ↓
Iteration 1:  Find confident docs → Update centroids
              ↓
Iteration 2:  Find confident docs → Update centroids
              ↓
Iteration 3:  Find confident docs → Update centroids
              ↓
Final:     Aligned class representations
```

### 3. Hierarchy Propagation Pattern

```
Leaf prediction:    [Baby Cereal]
                           ↓
Find ancestors:     [Baby Cereal] ← [Feeding] ← [Baby Product]
                           ↓
Expanded labels:    [Baby Cereal, Feeding, Baby Product]
```

## Module Dependencies

```
pipeline_teleclass.py
├── torch
├── transformers
│   ├── AutoTokenizer
│   ├── AutoModel
│   ├── BertForSequenceClassification
│   └── get_linear_schedule_with_warmup
├── sentence_transformers
│   └── SentenceTransformer
├── networkx (DiGraph)
├── pandas (CSV I/O)
├── numpy (array ops)
└── tqdm (progress bars)
```

## File I/O Flow

```
INPUT:
  Amazon_products/
    ├─ train/train_corpus.txt ──┐
    ├─ test/test_corpus.txt ────┤
    ├─ class_hierarchy.txt ─────┼──► DataLoader
    ├─ class_related_keywords.txt─┤
    └─ classes.txt ─────────────┘

INTERMEDIATE:
  outputs/intermediate/
    └─ phase2_outputs.pt
         ├─ pseudo_labels
         ├─ pseudo_scores
         ├─ class_embeddings
         └─ doc_embeddings

OUTPUT:
  outputs/
    ├─ models/
    │   ├─ best_model/ ────► For inference
    │   │   ├─ pytorch_model.bin
    │   │   ├─ config.json
    │   │   └─ tokenizer files
    │   └─ final_model/
    └─ submission.csv ────────► Upload to Kaggle
```

---

**Legend:**
- ★ = Critical innovation point
- → = Data flow
- ├─ = Sequential step
- └─ = Final output
- [time] = Approximate duration
