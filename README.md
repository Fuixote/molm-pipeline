# MOLM: Multi-Objective Learning Model for Antibody Sequence Co-Optimization

A multi-task deep learning framework for joint prediction of antibody affinity and specificity from binary deep-sequencing labels.

## Overview

MOLM jointly predicts affinity and specificity using a shared encoder with task-specific towers, trained on the [emibetuzumab co-optimization benchmark](https://www.nature.com/articles/s41467-022-31457-3) (Makowski et al., *Nature Communications*, 2022). The pipeline evaluates three feature representations (OneHot, ESM-2, Fusion-ESM2) across four evaluation paradigms: cross-validation, mutation-site holdout, cross-platform generalization, and Pareto-front recovery.

### Key Results

| Metric | MOLM | Best Baseline |
|--------|------|---------------|
| Holdout Affinity Accuracy | **91.24 ± 2.26%** | 90.36% (NN) |
| ISO Affinity Spearman ρ | **0.884** [0.84, 0.92] | 0.877 (NN) |
| Pareto Recall (latent PCA) | **0.47** | 0.20 (NN) |

## Architecture

```
Input (OneHot 2300D / ESM2 320D / Fusion-ESM2 2620D)
        │
  Shared Encoder (256 → 128, LayerNorm, GELU, Dropout 0.2)
        │
   ┌────┴────┐
   │         │
Affinity   Specificity
 Tower       Tower
(64→32)    (64→32)
   │         │
Latent 16D  Latent 16D  ──→ PCA → Pareto Recovery
   │         │
 Logit      Logit
   │         │
 ŷ_aff     ŷ_spec
```

**Training Objective** (per task): Focal Loss + Ranking Loss + Gap Hinge Loss  
**Optional**: Pareto-diversity regularizer (λ=0.1)

## Pipeline Structure

The pipeline is organized into 6 sequential phase files designed to run on Kaggle with T4 GPU:

| Phase | File | Description | Output | Runtime |
|-------|------|-------------|--------|---------|
| 0 | `phase0_config.py` | Configuration, model definitions, loss functions, metrics | — | Instant |
| 1 | `phase1_features.py` | Data loading, ESM-2 embedding computation | `features.pkl` | ~5 min |
| 2 | `phase2_baselines.py` | LDA + NN baselines (5-fold CV with AUC-ROC) | `baselines.pkl` | ~15 min |
| 3 | `phase3_molm_cv.py` | MOLM + MOLM-ST cross-validation (full grid) | `molm_cv.pkl` | ~3 hrs |
| 4 | `phase4_holdout.py` | Mutation-site holdout evaluation (8 sites × all models) | `holdout.pkl` | ~8 hrs |
| 5 | `phase5_generalization.py` | Cross-platform generalization + Pareto-front recovery | `generalization.pkl` | ~1.5 hrs |

## Setup & Usage

### Requirements

- Python 3.10+
- TensorFlow 2.x
- scikit-learn
- NumPy, Pandas
- `fair-esm` (for ESM-2 embeddings)

### Running on Kaggle

```python
# Cell 1: Load config
%run phase0_config.py

# Cell 2: Force ESM-2 to CPU (avoids CUDA kernel mismatch on Kaggle)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Cell 3: Compute features
%run phase1_features.py

# Cell 4: Restore GPU for training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Cell 5: Run baselines
%run phase2_baselines.py

# Cell 6: Run MOLM cross-validation
%run phase3_molm_cv.py

# Cell 7: Run holdout evaluation
%run phase4_holdout.py

# Cell 8: Free memory before generalization
import gc, tensorflow as tf
tf.keras.backend.clear_session()
gc.collect()

# Cell 9: Run generalization + Pareto analysis
%run phase5_generalization.py
```

### Dataset

The pipeline expects the [emibetuzumab dataset](https://www.nature.com/articles/s41467-022-31457-3) files in the Kaggle input directory:

- `emi_binding.csv` — EMI binary labels
- `emi_reps.csv` — EMI sequence representations
- `iso_binding.csv`, `iso_reps.csv` — ISO continuous measurements
- `igg_binding.csv`, `igg_reps.csv` — IgG continuous measurements
- `residue_dict.csv` — Residue mapping
- `emi_pl.txt`, `iso_pl.txt`, `igg_pl.txt` — Full sequences for ESM-2

## Feature Representations

| Feature | Dimensions | Source | Strengths |
|---------|-----------|--------|-----------|
| **OneHot** | 2,300D | Position × amino acid binary encoding | Best for mutation-site holdout extrapolation |
| **ESM-2** | 320D | Mean-pooled `esm2_t6_8M_UR50D` embeddings | Evolutionary context; poor alone for positional tasks |
| **Fusion-ESM2** | 2,620D | OneHot ∥ ESM-2 concatenation | Best for cross-platform generalization (ρ=0.884) |

## Evaluation Paradigms

1. **Cross-Validation**: 5-fold with joint-label (affinity × specificity) stratification
2. **Mutation-Site Holdout**: Train without top-performing residue at each CDR position, test on held-out sequences
3. **Cross-Platform Generalization**: Spearman ρ against independent ISO (yeast display, n=126) and IgG (soluble, n=42) measurements with bootstrap 95% CIs
4. **Pareto-Front Recovery**: Overlap between predicted and true Pareto-optimal variants from ISO continuous data

## Models Compared

| Model | Description | Parameters |
|-------|-------------|------------|
| **MOLM** | Multi-task shared encoder + dual towers | ~640K (OneHot) |
| **MOLM-ST** | Single-task, identical architecture to MOLM | ~640K (×2) |
| **NN** | Separate neural networks per task (Makowski et al. architecture) | ~48K (×2) |
| **LDA** | Linear Discriminant Analysis per task | ~2.3K (×2) |

## Citation

If you use this code, please cite:

```
@article{das2026molm,
  title={Multi-Objective Learning Model for Antibody Sequence Co-Optimization: 
         Generalization Under Sparse Binary Supervision},
  author={Das, Diganta and Yang, Haibo},
  journal={Bioinformatics},
  year={2026}
}
```

## References

- Makowski, E.K. et al. (2022). Co-optimization of therapeutic antibody affinity and specificity using machine learning models that generalize to novel mutational space. *Nature Communications*, 13:3788.
- Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637):1123–1130.
- Liu, L. et al. (2014). LY2875358, a Neutralizing and Internalizing Anti-MET Bivalent Antibody. *Clinical Cancer Research*, 20(23):6059–6070.

## License

MIT License
