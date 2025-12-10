# Quick Start Guide - TELEClass Pipeline

## Installation

1. **Install Dependencies**

```bash
cd taxoclass
pip install -r requirements_teleclass.txt
```

Required packages:
- PyTorch (2.0+)
- Transformers (4.30+)
- Sentence-Transformers (2.2+)
- NetworkX, Pandas, NumPy, scikit-learn, tqdm

## Pre-flight Check

Before running the full pipeline, test your setup:

```bash
python test_pipeline.py
```

This will verify:
- ✓ All packages are installed
- ✓ Data files exist and are readable
- ✓ CUDA is available (optional)
- ✓ Pipeline can be imported

Expected output:
```
============================================================
TELEClass Pipeline - Pre-flight Checks
============================================================
...
✓ All tests passed! Ready to run pipeline.
```

## Running the Pipeline

### Option 1: Basic Execution (Recommended)

```bash
python pipeline_teleclass.py
```

This runs the complete 6-phase pipeline with default settings.

### Option 2: Using Shell Script

```bash
./run_teleclass.sh
```

### Option 3: Custom Python Script

```python
from pipeline_teleclass import TELEClassPipeline

pipeline = TELEClassPipeline(
    data_dir="../Amazon_products",  # Adjust path as needed
    output_dir="outputs",
    seed=42
)
pipeline.run()
```

## Expected Runtime

| Phase | Task | Estimated Time (GPU) |
|-------|------|---------------------|
| 1 | Class & Document Encoding | ~5-10 min |
| 2 | Iterative Refinement | ~2-5 min |
| 3 | Augmentation | ~1 min |
| 4 | Hierarchy Expansion | ~1 min |
| 5 | BERT Training (3 epochs) | ~30-60 min |
| 6 | Inference | ~5-10 min |
| **Total** | | **~45-90 min** |

*Times assume NVIDIA GPU with 16GB+ VRAM*

## Output Files

After completion, check:

```
outputs/
├── models/
│   ├── best_model/              # Best model checkpoint
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── ...
│   └── final_model/             # Final model checkpoint
├── intermediate/
│   └── phase2_outputs.pt        # Pseudo-labels & embeddings
└── submission.csv               # Kaggle submission file
```

## Kaggle Submission

The final output is `outputs/submission.csv`:

```csv
ID,Label
0,grocery_gourmet_food beverages juices
1,toys_games games board_games
2,baby_product feeding baby_cereal
...
```

Upload this file directly to Kaggle.

## Examples

Run example scripts to learn advanced usage:

```bash
# Example 1: Basic pipeline
python example_teleclass.py 1

# Example 2: Custom hyperparameters
python example_teleclass.py 2

# Example 3: Incremental phases
python example_teleclass.py 3

# Example 4: Inference only
python example_teleclass.py 4

# Example 5: Analyze results
python example_teleclass.py 5

# Example 6: Train with validation split
python example_teleclass.py 6
```

## Customization

### Adjust Hyperparameters

Edit `pipeline_teleclass.py` main section:

```python
# Refinement parameters
labeler.refine_class_embeddings(
    doc_embeddings,
    class_embeddings,
    num_iterations=3,      # ← Increase for better refinement
    top_n_reliable=20,     # ← More documents = smoother centroids
)

# Training parameters
trainer.train(
    num_epochs=3,          # ← More epochs = better fit
    learning_rate=2e-5,    # ← Adjust for convergence
    batch_size=16          # ← Adjust for memory
)

# Inference threshold
predictions = inference.predict(
    texts,
    threshold=0.5          # ← Lower = more labels
)
```

### Key Parameters to Tune

| Parameter | Default | Effect | Recommendation |
|-----------|---------|--------|----------------|
| `num_iterations` | 3 | Refinement quality | 2-5 |
| `top_n_reliable` | 20 | Centroid stability | 10-50 |
| `min_labels` | 2 | Label coverage | 2-5 |
| `num_epochs` | 3 | Model fit | 3-5 |
| `learning_rate` | 2e-5 | Training speed | 1e-5 to 5e-5 |
| `threshold` | 0.5 | Prediction sensitivity | 0.3-0.7 |

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch sizes
trainer.prepare_data(..., batch_size=8)  # Default: 16
inference.predict(..., batch_size=16)    # Default: 32

# Reduce sequence length
trainer.prepare_data(..., max_length=64)  # Default: 128
```

### Poor Performance on Kaggle

1. **Increase refinement iterations:**
   ```python
   num_iterations=5  # Default: 3
   ```

2. **Add more training epochs:**
   ```python
   num_epochs=5  # Default: 3
   ```

3. **Lower inference threshold:**
   ```python
   threshold=0.3  # Default: 0.5
   ```

4. **Implement LLM augmentation** for starved classes (see `AugmentationModule`)

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements_teleclass.txt

# If sentence-transformers fails
pip install sentence-transformers --no-cache-dir
```

### Data Not Found

The pipeline looks for data in:
1. `Amazon_products/` (if running from project root)
2. `../Amazon_products/` (if running from `taxoclass/` directory)

Verify structure:
```bash
ls -la ../Amazon_products/
# Should show: train/, test/, class_hierarchy.txt, etc.
```

## Advanced: Resume from Checkpoint

If training is interrupted, you can resume from intermediate results:

```python
# Load Phase 2 outputs
results = torch.load("outputs/intermediate/phase2_outputs.pt")
pseudo_labels = results['pseudo_labels']
doc_embeddings = results['doc_embeddings']

# Continue from Phase 4 (skip 1-3)
expander = HierarchyExpander(...)
expanded_labels = expander.expand_labels_with_hierarchy(pseudo_labels)

# Continue with training...
```

## Performance Tips

1. **Use GPU:** Ensure CUDA is available (check with `test_pipeline.py`)
2. **Increase batch sizes** if you have more VRAM
3. **Use mixed precision training** (add to trainer)
4. **Parallelize encoding** (sentence-transformers supports multi-GPU)

## Next Steps

- Read `README_TELECLASS.md` for detailed documentation
- Check `example_teleclass.py` for advanced usage patterns
- Experiment with hyperparameters using `example_teleclass.py 2`
- Analyze results with `example_teleclass.py 5`

## Support

For issues or questions:
1. Check test output: `python test_pipeline.py`
2. Review logs in console output
3. Examine intermediate results in `outputs/intermediate/`
4. Verify data format matches expected structure

## Research Citation

This pipeline is based on the TELEClass framework. If you use it in research:

```bibtex
@inproceedings{zhang2021teleclass,
  title={TELEClass: Taxonomy Enrichment and LLM-Enhanced Hierarchical Text Classification},
  author={Zhang, Yue and Yu, Yinghao and Wang, Jiaming},
  booktitle={EMNLP},
  year={2021}
}
```
