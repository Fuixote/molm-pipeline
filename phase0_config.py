"""
MOLM Pipeline — Phase 0: Config & Shared Utilities
====================================================
Contains all imports, Config, model classes, loss functions, metrics,
and utility functions shared across all phases.

Usage: Every other phase does `from phase0_config import *`
"""

# ============================================================================
# DETERMINISTIC SETUP — MUST BE AT VERY TOP
# ============================================================================
import os
import random

os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ============================================================================
# IMPORTS
# ============================================================================
import gc
gc.collect()

try:
    import tensorflow as tf
    tf.keras.backend.clear_session()
    del tf
except:
    pass

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate as cv
from sklearn.model_selection import cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import stats
from scipy.stats import binomtest
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import warnings
import json
import sys
import time
import hashlib
import pickle

# ============================================================================
# DETERMINISTIC SEEDING
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except:
    pass

tf.keras.backend.clear_session()
gc.collect()

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# GPU SETUP
# ============================================================================
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass

tf.keras.mixed_precision.set_global_policy("float32")
tf.config.optimizer.set_jit(False)

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    DATA_PATH = "/kaggle/input/datasets/iamdiganta7/antibody"
    OUTPUT_DIR = "/kaggle/working/molm_pipeline_results"
    
    RUN_LDA_BASELINES = True
    RUN_NN_BASELINES = True
    RUN_MOLM = True
    RUN_MOLM_ST = True
    
    DEBUG_MODE = True
    SAVE_FOLD_MANIFEST = True
    SAVE_EPOCH_CSV = True
    SAVE_PAPER_FIG = True
    PAPER_FIG_NAME = "paper_diag.png"
    GEN_EVAL_EVERY = 5
    
    SEQ_LENGTH = 115
    MUTATION_SITES = [32, 49, 54, 55, 56, 98, 100, 103]
    MUTATION_SITES_KABAT = [33, 50, 54, 55, 56, 95, 97, 102]
    WILDTYPE_RESIDUES = ['Y', 'R', 'R', 'R', 'G', 'A', 'W', 'Y']
    TOP_RESIDUES = ['V', 'E', 'G', 'G', 'D', 'S', 'L', 'D']
    SAMPLED_RESIDUES = [
        'YFVASD', 'RKGATEM', 'RKGATE', 'RKGATE',
        'GANSTD', 'AFVYSD', 'WLVGAS', 'YFVASDT',
    ]
    
    SHARED_DIMS = [256, 128]
    TOWER_DIMS = [64, 32]
    LATENT_DIM = 16
    DROPOUT_RATE = 0.2
    
    EPOCHS = 25
    BATCH_SIZE = 64
    CV_FOLDS = 5
    MC_SAMPLES = 50
    LEARNING_RATE = 5e-5
    
    ADVERSARIAL_WEIGHT = 0.0
    ORTHO_WEIGHT = 0.0
    ADV_WARMUP_EPOCHS = 10
    
    RANKING_WEIGHT_AFF = 0.3
    RANKING_WEIGHT_SPEC = 0.6
    GAP_WEIGHT_AFF = 0.2
    GAP_WEIGHT_SPEC = 0.6
    FOCAL_GAMMA = 2.0
    GRL_LAMBDA = 1.0
    RANKING_MARGIN = 0.3
    GAP_MARGIN = 0.2
    
    FEATURE_TYPES = ['onehot', 'esm2', 'fusion_esm2']
    GRID_FEATURE_TYPES = ['onehot', 'esm2', 'fusion_esm2']
    
    ESM2_DIR = "/kaggle/working"
    ESM2_DIM = 320
    USE_ESM2 = True
    
    COMPUTE_AUC = True
    BOOTSTRAP_N = 1000
    
    # === PARETO EXPERIMENT FLAGS ===
    PARETO_LOSS = False           # Option 3: Add Pareto-diversity loss during training
    PARETO_LOSS_WEIGHT = 0.1     # Moderate weight — acts as regularizer
    PARETO_LOSS_WARMUP = 5       # Don't apply until epoch 5 (let classification stabilize)
    PARETO_SCORE_TYPES = ['logits', 'probs', 'latent_pca']  # Option 2: Evaluate all 3

config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# Feature label mapping (used across all phases)
FEAT_LABELS = {
    'onehot': 'OneHot', 'esm2': 'ESM2',
    'fusion_esm2': 'Fusion-ESM2', 'fusion': 'Fusion-UniRep'
}

def get_primary_feat(features):
    """Determine primary feature type."""
    if 'fusion_esm2' in features.get('emi', {}):
        return 'fusion_esm2', 'Fusion-ESM2'
    return 'fusion', 'Fusion-UniRep'

# ============================================================================
# LOGGING
# ============================================================================
class DebugLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.logs = []
        with open(log_path, 'w') as f:
            f.write(f"# Debug Log - {datetime.now().isoformat()}\n{'='*70}\n\n")
    
    def log(self, msg, to_console=True):
        ts = datetime.now().strftime("%H:%M:%S")
        fmt = f"[{ts}] {msg}"
        self.logs.append(fmt)
        with open(self.log_path, 'a') as f:
            f.write(fmt + "\n")
        if to_console:
            print(fmt)
    
    def log_json(self, data, label=""):
        self.log(f"{label}:\n{json.dumps(data, indent=2, default=str)}", to_console=config.DEBUG_MODE)
    
    def log_separator(self, title=""):
        sep = "-" * 50
        self.log(f"\n{sep}\n{title}\n{sep}" if title else sep)

logger = DebugLogger(os.path.join(config.OUTPUT_DIR, "debug_log.txt"))

# ============================================================================
# DETERMINISTIC UTILITIES
# ============================================================================
def hard_reset_rng(seed, context=""):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.keras.backend.clear_session()
    gc.collect()
    if config.DEBUG_MODE:
        logger.log(f"🔄 RNG Reset: seed={seed} ({context})")

def hash_indices(indices):
    return hashlib.md5(np.array(indices).tobytes()).hexdigest()[:12]

def get_index_fingerprint(indices):
    indices = np.array(indices)
    return {'count': len(indices), 'hash': hash_indices(indices),
            'head_10': indices[:10].tolist(), 'tail_10': indices[-10:].tolist(),
            'min': int(indices.min()), 'max': int(indices.max()), 'sum': int(indices.sum())}

# ============================================================================
# JOINT LABEL STRATIFICATION
# ============================================================================
def create_joint_labels(y_aff, y_spec):
    return y_aff * 2 + y_spec

def get_stratified_kfold(y_aff, y_spec, n_splits=5, random_state=42):
    y_joint = create_joint_labels(y_aff, y_spec)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return skf, y_joint

def print_joint_label_distribution(y_aff, y_spec, prefix=""):
    y_joint = create_joint_labels(y_aff, y_spec)
    labels = ['(Aff=0,Spec=0)', '(Aff=0,Spec=1)', '(Aff=1,Spec=0)', '(Aff=1,Spec=1)']
    print(f"{prefix}Joint label distribution:")
    for i, label in enumerate(labels):
        count = (y_joint == i).sum()
        print(f"{prefix}  {label}: {count:4d} ({100*count/len(y_joint):5.1f}%)")

# ============================================================================
# FEATURE GENERATION
# ============================================================================
ALPH_LETTERS = np.array(sorted('ACDEFGHIKLMNPQRSTVWY'))
_le = LabelEncoder()
_integer_encoded_letters = _le.fit_transform(ALPH_LETTERS).reshape(-1, 1)
_one = OneHotEncoder(sparse_output=False)
_ohe_letters = _one.fit_transform(_integer_encoded_letters)

def generate_onehot(seqs_binding):
    ohe = []
    for seq in seqs_binding.index:
        chars = _le.transform(list(seq))
        let = chars.reshape(config.SEQ_LENGTH, 1)
        ohe.append(_one.transform(let).flatten())
    return pd.DataFrame(np.stack(ohe))

# ============================================================================
# HELD-OUT UTILITIES
# ============================================================================
def get_sequences_with_residue(sequences, site_idx, residue):
    return [s for s in sequences if len(s) > site_idx and s[site_idx] == residue]

def get_sequences_without_residue(sequences, site_idx, residue):
    return [s for s in sequences if len(s) > site_idx and s[site_idx] != residue]

def create_holdout_indices(sequences, site_idx, residue, mode='top'):
    seq_list = list(sequences)
    if mode == 'top':
        test_seqs = get_sequences_with_residue(sequences, site_idx, residue)
        train_seqs = get_sequences_without_residue(sequences, site_idx, residue)
    else:
        train_seqs = get_sequences_with_residue(sequences, site_idx, residue)
        test_seqs = get_sequences_without_residue(sequences, site_idx, residue)
    train_idx = np.array([seq_list.index(s) for s in train_seqs])
    test_idx = np.array([seq_list.index(s) for s in test_seqs])
    return train_idx, test_idx

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
def focal_bce_with_logits(y_true, logit, gamma=2.0, pos_weight=1.0):
    y_true = tf.cast(y_true, tf.float32)
    logit = tf.cast(logit, tf.float32)
    bce = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=logit, pos_weight=pos_weight)
    p = tf.sigmoid(logit)
    p_t = y_true * p + (1 - y_true) * (1 - p)
    return tf.reduce_mean(tf.pow(1.0 - p_t, gamma) * bce)

def safe_gap(scores, labels):
    scores = tf.cast(scores, tf.float32)
    labels = tf.cast(labels, tf.float32)
    pos_mask = labels > 0.5
    neg_mask = labels < 0.5
    n_pos = tf.reduce_sum(tf.cast(pos_mask, tf.float32))
    n_neg = tf.reduce_sum(tf.cast(neg_mask, tf.float32))
    def compute():
        return tf.reduce_mean(tf.boolean_mask(scores, pos_mask)) - tf.reduce_mean(tf.boolean_mask(scores, neg_mask))
    return tf.cond(tf.logical_and(n_pos > 0, n_neg > 0), compute, lambda: tf.constant(0.0, tf.float32))

def gap_hinge_loss(scores, labels, margin=0.2):
    scores = tf.cast(scores, tf.float32)
    labels = tf.cast(labels, tf.float32)
    pos_mask = labels > 0.5
    neg_mask = labels < 0.5
    n_pos = tf.reduce_sum(tf.cast(pos_mask, tf.float32))
    n_neg = tf.reduce_sum(tf.cast(neg_mask, tf.float32))
    def compute():
        gap = tf.reduce_mean(tf.boolean_mask(scores, pos_mask)) - tf.reduce_mean(tf.boolean_mask(scores, neg_mask))
        return tf.nn.relu(margin - gap)
    return tf.cond(tf.logical_and(n_pos > 0, n_neg > 0), compute, lambda: 0.0)

def ranking_loss(scores, labels, margin=0.3):
    scores = tf.cast(scores, tf.float32)
    labels = tf.cast(labels, tf.float32)
    pos_mask = labels > 0.5
    neg_mask = labels < 0.5
    n_pos = tf.reduce_sum(tf.cast(pos_mask, tf.float32))
    n_neg = tf.reduce_sum(tf.cast(neg_mask, tf.float32))
    def compute():
        pos_sc = tf.boolean_mask(scores, pos_mask)
        neg_sc = tf.boolean_mask(scores, neg_mask)
        diff = tf.expand_dims(pos_sc, 1) - tf.expand_dims(neg_sc, 0)
        return tf.reduce_mean(tf.maximum(0.0, margin - diff))
    return tf.cond(tf.logical_and(n_pos > 0, n_neg > 0), compute, lambda: 0.0)

def adversarial_loss(disc_pred_aff, disc_pred_spec):
    loss_aff = tf.keras.losses.binary_crossentropy(tf.ones_like(disc_pred_aff), disc_pred_aff)
    loss_spec = tf.keras.losses.binary_crossentropy(tf.zeros_like(disc_pred_spec), disc_pred_spec)
    return tf.reduce_mean(loss_aff) + tf.reduce_mean(loss_spec)

def independence_loss(z1, z2):
    z1 = tf.cast(z1, tf.float32); z2 = tf.cast(z2, tf.float32)
    z1_n = (z1 - tf.reduce_mean(z1, 0, True)) / (tf.math.reduce_std(z1, 0, True) + 1e-6)
    z2_n = (z2 - tf.reduce_mean(z2, 0, True)) / (tf.math.reduce_std(z2, 0, True) + 1e-6)
    corr = tf.matmul(z1_n, z2_n, transpose_a=True) / tf.cast(tf.shape(z1)[0], tf.float32)
    return tf.reduce_mean(tf.square(corr))

def pareto_diversity_loss(aff_scores, spec_scores):
    """
    Pareto-diversity loss: Encourages the model to spread predictions
    across the affinity-specificity tradeoff surface instead of clustering.
    
    Three components:
    1. SPREAD: Maximize variance of predictions in both dimensions
    2. DECORRELATION: Penalize positive correlation between task scores
       (high corr = all predictions in one corner, not along the frontier)
    3. FRONTIER COVERAGE: Encourage predictions at different tradeoff ratios
    
    Returns: scalar loss to MINIMIZE (higher = worse Pareto spread)
    """
    aff_p = tf.sigmoid(tf.cast(aff_scores, tf.float32))
    spec_p = tf.sigmoid(tf.cast(spec_scores, tf.float32))
    
    # 1. Spread: penalize low variance in either dimension
    spread_aff = tf.math.reduce_std(aff_p)
    spread_spec = tf.math.reduce_std(spec_p)
    spread_loss = -tf.math.log(spread_aff + 1e-6) - tf.math.log(spread_spec + 1e-6)
    
    # 2. Decorrelation: penalize positive correlation between scores
    # We want: some seqs high-aff/low-spec, some low-aff/high-spec
    aff_c = aff_p - tf.reduce_mean(aff_p)
    spec_c = spec_p - tf.reduce_mean(spec_p)
    corr = tf.reduce_sum(aff_c * spec_c) / (
        tf.norm(aff_c) * tf.norm(spec_c) + 1e-8)
    decorr_loss = tf.nn.relu(corr)  # Only penalize POSITIVE correlation
    
    # 3. Frontier coverage: encourage diverse tradeoff ratios
    # Angle in (aff, spec) space — want uniform spread of angles
    angles = tf.atan2(spec_p - tf.reduce_mean(spec_p), aff_p - tf.reduce_mean(aff_p))
    angle_std = tf.math.reduce_std(angles)
    coverage_loss = -tf.math.log(angle_std + 1e-6)
    
    return spread_loss + decorr_loss + 0.5 * coverage_loss

def tfp_correlation(x, y):
    x = x - tf.reduce_mean(x); y = y - tf.reduce_mean(y)
    return tf.reduce_sum(x * y) / (tf.norm(x) * tf.norm(y) + 1e-8)

# ============================================================================
# METRIC UTILITIES
# ============================================================================
def safe_auc(y_true, y_prob, metric='roc'):
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return roc_auc_score(y_true, y_prob) if metric == 'roc' else average_precision_score(y_true, y_prob)

def safe_mcc(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float('nan')
    return matthews_corrcoef(y_true, y_pred)

def bootstrap_spearman_ci(x, y, n_bootstrap=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    rhos = []
    n = len(x)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        rho, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(rho):
            rhos.append(rho)
    rhos = np.sort(rhos)
    alpha = (1 - ci) / 2
    return np.percentile(rhos, 100 * alpha), np.percentile(rhos, 100 * (1 - alpha))

def mcnemar_test(y_true, pred_a, pred_b):
    correct_a = (pred_a == y_true).astype(int)
    correct_b = (pred_b == y_true).astype(int)
    n01 = np.sum((correct_a == 1) & (correct_b == 0))
    n10 = np.sum((correct_a == 0) & (correct_b == 1))
    n_discord = n01 + n10
    if n_discord == 0:
        return 0.0, 1.0
    if n_discord < 25:
        try:
            result = binomtest(n01, n_discord, 0.5)
            p_val = result.pvalue
        except Exception:
            chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
            p_val = stats.chi2.sf(chi2, df=1)
    else:
        chi2 = (n01 - n10) ** 2 / (n01 + n10)
        p_val = stats.chi2.sf(chi2, df=1)
    return float(n01 - n10), float(p_val)

def compute_pareto_front(aff_scores, spec_scores):
    n = len(aff_scores)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j: continue
            if (aff_scores[j] >= aff_scores[i] and spec_scores[j] >= spec_scores[i] and
                (aff_scores[j] > aff_scores[i] or spec_scores[j] > spec_scores[i])):
                is_pareto[i] = False
                break
    return is_pareto

# ============================================================================
# DIAGNOSTIC STORAGE
# ============================================================================
@dataclass
class EpochDiagnostics:
    epoch: int
    loss_total: float = 0.0
    loss_cls_aff: float = 0.0; loss_cls_spec: float = 0.0
    loss_rank_aff: float = 0.0; loss_rank_spec: float = 0.0
    loss_gap_aff: float = 0.0; loss_gap_spec: float = 0.0
    loss_adv: float = 0.0; loss_indep: float = 0.0
    acc_aff: float = 0.0; acc_spec: float = 0.0
    aff_score_mean: float = 0.0; aff_score_std: float = 0.0
    spec_score_mean: float = 0.0; spec_score_std: float = 0.0
    aff_gap: float = 0.0; spec_gap: float = 0.0
    aff_latent_mean: float = 0.0; aff_latent_std: float = 0.0
    spec_latent_mean: float = 0.0; spec_latent_std: float = 0.0
    latent_correlation: float = 0.0
    dead_neurons_aff: int = 0; dead_neurons_spec: int = 0
    grad_norm_shared: float = 0.0
    grad_norm_aff_tower: float = 0.0; grad_norm_spec_tower: float = 0.0
    grad_norm_aff_head: float = 0.0; grad_norm_spec_head: float = 0.0
    grad_cosine_aff_spec: float = 0.0
    disc_acc: float = 0.0
    iso_aff_rho: Optional[float] = None; iso_spec_rho: Optional[float] = None
    igg_aff_rho: Optional[float] = None; igg_spec_rho: Optional[float] = None

# ============================================================================
# MODEL: GRL + MCDropout
# ============================================================================
@tf.custom_gradient
def gradient_reversal_func(x, lambd):
    def grad(dy):
        return -tf.cast(lambd, dy.dtype) * dy, None
    return x, grad

@tf.keras.utils.register_keras_serializable()
class GradientReversalLayer(layers.Layer):
    def __init__(self, lambd=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambd = lambd
    def call(self, x):
        return gradient_reversal_func(x, self.lambd)
    def get_config(self):
        base = super().get_config()
        base['lambd'] = self.lambd
        return base

@tf.keras.utils.register_keras_serializable()
class MCDropout(layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=training)

# ============================================================================
# MODEL: DiagnosticMOLM
# ============================================================================
@tf.keras.utils.register_keras_serializable()
class DiagnosticMOLM(Model):
    def __init__(self, input_dim, latent_dim=16, shared_dims=[256, 128],
                 tower_dims=[64, 32], dropout_rate=0.2, grl_lambda=1.0, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.grl_lambda = grl_lambda
        self._shared_dims = list(shared_dims)
        self._tower_dims = list(tower_dims)
        self._dropout_rate = dropout_rate
        
        self.shared_layers = []
        for i, dim in enumerate(shared_dims):
            self.shared_layers.append(layers.Dense(dim, name=f'shared_dense_{i}'))
            self.shared_layers.append(layers.LayerNormalization(name=f'shared_ln_{i}'))
            self.shared_layers.append(MCDropout(dropout_rate, name=f'shared_drop_{i}'))
        
        self.grl = GradientReversalLayer(lambd=grl_lambda)
        self.disc_dense1 = layers.Dense(32, activation='gelu', name='disc_dense1')
        self.disc_dense2 = layers.Dense(16, activation='gelu', name='disc_dense2')
        self.disc_out = layers.Dense(1, activation='sigmoid', name='disc_out', dtype='float32')
        
        self.aff_layers = []
        for i, dim in enumerate(tower_dims):
            self.aff_layers.append(layers.Dense(dim, name=f'aff_dense_{i}'))
            self.aff_layers.append(layers.LayerNormalization(name=f'aff_ln_{i}'))
            self.aff_layers.append(MCDropout(dropout_rate, name=f'aff_drop_{i}'))
        self.aff_proj = layers.Dense(latent_dim, name='aff_proj')
        self.aff_proj_norm = layers.LayerNormalization(name='aff_proj_norm')
        
        self.spec_layers = []
        for i, dim in enumerate(tower_dims):
            self.spec_layers.append(layers.Dense(dim, name=f'spec_dense_{i}'))
            self.spec_layers.append(layers.LayerNormalization(name=f'spec_ln_{i}'))
            self.spec_layers.append(MCDropout(dropout_rate, name=f'spec_drop_{i}'))
        self.spec_proj = layers.Dense(latent_dim, name='spec_proj')
        self.spec_proj_norm = layers.LayerNormalization(name='spec_proj_norm')
        
        self.aff_head = layers.Dense(1, name='aff_head', dtype='float32')
        self.spec_head = layers.Dense(1, name='spec_head', dtype='float32')
    
    def run_shared(self, x, training):
        for i in range(0, len(self.shared_layers), 3):
            x = self.shared_layers[i](x)
            x = self.shared_layers[i+1](x)
            x = tf.nn.gelu(x)
            x = self.shared_layers[i+2](x, training=training)
        return x
    
    def run_tower(self, x, tower_layers, proj, proj_norm, training):
        for i in range(0, len(tower_layers), 3):
            x = tower_layers[i](x)
            x = tower_layers[i+1](x)
            x = tf.nn.gelu(x)
            x = tower_layers[i+2](x, training=training)
        return proj_norm(proj(x))
    
    def run_discriminator(self, x):
        return self.disc_out(self.disc_dense2(self.disc_dense1(tf.cast(x, tf.float32))))
    
    def call(self, inputs, training=None):
        shared = self.run_shared(inputs, training)
        aff_latent = self.run_tower(shared, self.aff_layers, self.aff_proj, self.aff_proj_norm, training)
        spec_latent = self.run_tower(shared, self.spec_layers, self.spec_proj, self.spec_proj_norm, training)
        aff_logit = tf.squeeze(self.aff_head(aff_latent), -1)
        spec_logit = tf.squeeze(self.spec_head(spec_latent), -1)
        return {
            'aff_score': aff_logit, 'spec_score': spec_logit,
            'aff_prob': tf.sigmoid(tf.cast(aff_logit, tf.float32)),
            'spec_prob': tf.sigmoid(tf.cast(spec_logit, tf.float32)),
            'aff_latent': aff_latent, 'spec_latent': spec_latent,
            'aff_latent_grl': self.grl(aff_latent),
            'spec_latent_grl': self.grl(spec_latent),
            'shared': shared,
        }
    
    def get_config(self):
        base = super().get_config()
        base.update({'input_dim': self.input_dim, 'latent_dim': self.latent_dim,
                     'grl_lambda': self.grl_lambda, 'shared_dims': self._shared_dims,
                     'tower_dims': self._tower_dims, 'dropout_rate': self._dropout_rate})
        return base
    
    def get_layer_groups(self):
        return {
            'shared': [l for l in self.shared_layers if hasattr(l, 'trainable_weights')],
            'aff_tower': self.aff_layers + [self.aff_proj, self.aff_proj_norm],
            'spec_tower': self.spec_layers + [self.spec_proj, self.spec_proj_norm],
            'aff_head': [self.aff_head], 'spec_head': [self.spec_head],
            'discriminator': [self.disc_dense1, self.disc_dense2, self.disc_out],
        }
    
    def get_projections(self, inputs):
        out = self(inputs, training=False)
        return {'affinity': out['aff_score'].numpy().flatten(),
                'specificity': out['spec_score'].numpy().flatten()}

# ============================================================================
# TRAINER: DiagnosticTrainer (Multi-Task)
# ============================================================================
class DiagnosticTrainer:
    def __init__(self, model, aff_pos_weight=1.0, spec_pos_weight=1.0, learning_rate=5e-5):
        self.model = model
        self.aff_pos_weight = aff_pos_weight
        self.spec_pos_weight = spec_pos_weight
        self.optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4, clipnorm=1.0)
        self.epoch_diagnostics = []
        self.history = defaultdict(list)
    
    def train_step_with_diagnostics(self, x, y_aff, y_spec):
        with tf.GradientTape(persistent=True) as tape:
            out = self.model(x, training=True)
            loss_cls_aff = focal_bce_with_logits(y_aff, out['aff_score'], config.FOCAL_GAMMA, self.aff_pos_weight)
            loss_cls_spec = focal_bce_with_logits(y_spec, out['spec_score'], config.FOCAL_GAMMA, self.spec_pos_weight)
            loss_rank_aff = ranking_loss(out['aff_score'], y_aff, config.RANKING_MARGIN)
            loss_rank_spec = ranking_loss(out['spec_score'], y_spec, config.RANKING_MARGIN)
            loss_gap_aff = gap_hinge_loss(out['aff_score'], y_aff, config.GAP_MARGIN)
            loss_gap_spec = gap_hinge_loss(out['spec_score'], y_spec, config.GAP_MARGIN)
            disc_on_aff = self.model.run_discriminator(out['aff_latent_grl'])
            disc_on_spec = self.model.run_discriminator(out['spec_latent_grl'])
            loss_adv = adversarial_loss(disc_on_aff, disc_on_spec)
            loss_indep = independence_loss(out['aff_latent'], out['spec_latent'])
            total = (loss_cls_aff + loss_cls_spec
                    + config.RANKING_WEIGHT_AFF * loss_rank_aff + config.RANKING_WEIGHT_SPEC * loss_rank_spec
                    + config.GAP_WEIGHT_AFF * loss_gap_aff + config.GAP_WEIGHT_SPEC * loss_gap_spec
                    + config.ADVERSARIAL_WEIGHT * loss_adv + config.ORTHO_WEIGHT * loss_indep)
            
            # Pareto diversity loss (Option 3 — gated by config flag + warmup)
            loss_pareto = tf.constant(0.0, tf.float32)
            if config.PARETO_LOSS:
                loss_pareto = pareto_diversity_loss(out['aff_score'], out['spec_score'])
                total = total + config.PARETO_LOSS_WEIGHT * loss_pareto
        
        grads = tape.gradient(total, self.model.trainable_variables)
        grad_aff = tape.gradient(loss_cls_aff, self.model.trainable_variables)
        grad_spec = tape.gradient(loss_cls_spec, self.model.trainable_variables)
        del tape
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        def flatten_grads(g):
            flat = [tf.reshape(x, [-1]) for x in g if x is not None]
            return tf.concat(flat, 0) if flat else tf.zeros([1])
        cosine_sim = tf.reduce_sum(flatten_grads(grad_aff) * flatten_grads(grad_spec)) / (
            tf.norm(flatten_grads(grad_aff)) * tf.norm(flatten_grads(grad_spec)) + 1e-8)
        
        acc_aff = tf.reduce_mean(tf.cast((out['aff_score'] > 0) == tf.cast(y_aff, tf.bool), tf.float32))
        acc_spec = tf.reduce_mean(tf.cast((out['spec_score'] > 0) == tf.cast(y_spec, tf.bool), tf.float32))
        
        return {
            'loss_total': float(total), 'loss_cls_aff': float(loss_cls_aff), 'loss_cls_spec': float(loss_cls_spec),
            'loss_rank_aff': float(loss_rank_aff), 'loss_rank_spec': float(loss_rank_spec),
            'loss_gap_aff': float(loss_gap_aff), 'loss_gap_spec': float(loss_gap_spec),
            'loss_adv': float(loss_adv), 'loss_indep': float(loss_indep),
            'acc_aff': float(acc_aff), 'acc_spec': float(acc_spec),
            'aff_gap': float(safe_gap(out['aff_score'], y_aff)),
            'spec_gap': float(safe_gap(out['spec_score'], y_spec)),
            'latent_correlation': float(tf.abs(tfp_correlation(tf.reshape(out['aff_latent'], [-1]),
                                                                tf.reshape(out['spec_latent'], [-1])))),
            'dead_neurons_aff': int(tf.reduce_sum(tf.cast(tf.math.reduce_variance(out['aff_latent'], 0) < 1e-6, tf.int32))),
            'dead_neurons_spec': int(tf.reduce_sum(tf.cast(tf.math.reduce_variance(out['spec_latent'], 0) < 1e-6, tf.int32))),
            'grad_cosine_aff_spec': float(cosine_sim),
            'aff_score_mean': float(tf.reduce_mean(out['aff_score'])),
            'spec_score_mean': float(tf.reduce_mean(out['spec_score'])),
            'loss_pareto': float(loss_pareto),
        }
    
    def create_dataset(self, X, y_aff, y_spec, batch_size, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices((tf.cast(X, tf.float32), tf.cast(y_aff, tf.int64), tf.cast(y_spec, tf.int64)))
        if shuffle:
            ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=False)
        opts = tf.data.Options(); opts.deterministic = True
        return ds.with_options(opts).prefetch(1)
    
    def fit(self, X, y_aff, y_spec, epochs=25, batch_size=64,
            X_iso=None, y_iso_aff=None, y_iso_spec=None,
            X_igg=None, y_igg_aff=None, y_igg_spec=None, verbose=1):
        dataset = self.create_dataset(X, y_aff, y_spec, batch_size)
        _pareto_orig = config.PARETO_LOSS  # Save original flag
        for epoch in range(epochs):
            # Pareto warmup: don't apply until training stabilizes
            if _pareto_orig and epoch < config.PARETO_LOSS_WARMUP:
                config.PARETO_LOSS = False
            elif _pareto_orig:
                config.PARETO_LOSS = True
            
            epoch_metrics = defaultdict(list)
            for x_b, ya_b, ys_b in dataset:
                bm = self.train_step_with_diagnostics(x_b, ya_b, ys_b)
                for k, v in bm.items():
                    epoch_metrics[k].append(v)
            
            diag = EpochDiagnostics(epoch=epoch)
            for k in epoch_metrics:
                if hasattr(diag, k):
                    setattr(diag, k, np.mean(epoch_metrics[k]))
            
            if (epoch + 1) % config.GEN_EVAL_EVERY == 0 or epoch == epochs - 1:
                if X_iso is not None:
                    out_iso = self.model(tf.constant(X_iso, tf.float32), training=False)
                    diag.iso_aff_rho, _ = stats.spearmanr(out_iso['aff_score'].numpy(), y_iso_aff)
                    diag.iso_spec_rho, _ = stats.spearmanr(out_iso['spec_score'].numpy(), y_iso_spec)
                if X_igg is not None:
                    out_igg = self.model(tf.constant(X_igg, tf.float32), training=False)
                    diag.igg_aff_rho, _ = stats.spearmanr(out_igg['aff_score'].numpy()[:42], y_igg_aff[:42])
                    diag.igg_spec_rho, _ = stats.spearmanr(out_igg['spec_score'].numpy()[:42], y_igg_spec[:42])
            
            self.epoch_diagnostics.append(diag)
            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(f"  Ep {epoch+1:2d}/{epochs} | Acc: {diag.acc_aff:.3f}/{diag.acc_spec:.3f} | "
                      f"Gap: {diag.aff_gap:+.2f}/{diag.spec_gap:+.2f} | GradCos: {diag.grad_cosine_aff_spec:+.3f}")
        config.PARETO_LOSS = _pareto_orig  # Restore original flag
        return self.history

# ============================================================================
# TRAINER: MOLMSingleTaskTrainer (MOLM-ST)
# ============================================================================
class MOLMSingleTaskTrainer:
    def __init__(self, model, task='affinity', aff_pos_weight=1.0, spec_pos_weight=1.0,
                 learning_rate=5e-5, ranking_weight=0.3, gap_weight=0.2,
                 focal_gamma=2.0, ranking_margin=0.3, gap_margin=0.2):
        self.model = model; self.task = task
        self.aff_pos_weight = aff_pos_weight; self.spec_pos_weight = spec_pos_weight
        self.focal_gamma = focal_gamma; self.ranking_margin = ranking_margin
        self.gap_margin = gap_margin; self.ranking_weight = ranking_weight; self.gap_weight = gap_weight
        self.optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4, clipnorm=1.0)
    
    @tf.function
    def train_step(self, x, y_aff, y_spec):
        with tf.GradientTape() as tape:
            out = self.model(x, training=True)
            score = out['aff_score'] if self.task == 'affinity' else out['spec_score']
            labels = y_aff if self.task == 'affinity' else y_spec
            pw = self.aff_pos_weight if self.task == 'affinity' else self.spec_pos_weight
            total = (focal_bce_with_logits(labels, score, self.focal_gamma, pw)
                     + self.ranking_weight * ranking_loss(score, labels, self.ranking_margin)
                     + self.gap_weight * gap_hinge_loss(score, labels, self.gap_margin))
        self.optimizer.apply_gradients(zip(tape.gradient(total, self.model.trainable_variables), self.model.trainable_variables))
        return total
    
    def fit(self, X, y_aff, y_spec, epochs=25, batch_size=64, verbose=0):
        ds = tf.data.Dataset.from_tensor_slices((tf.cast(X, tf.float32), tf.cast(y_aff, tf.int64), tf.cast(y_spec, tf.int64)))
        ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True).batch(batch_size)
        opts = tf.data.Options(); opts.deterministic = True
        ds = ds.with_options(opts).prefetch(1)
        for epoch in range(epochs):
            for x_b, ya_b, ys_b in ds:
                self.train_step(x_b, ya_b, ys_b)

def train_molm_st(X, y_aff, y_spec, task, config_obj, aff_pos_weight=1.0, spec_pos_weight=1.0, seed_offset=0):
    hard_reset_rng(SEED + seed_offset, f"MOLM-ST {task}")
    model = DiagnosticMOLM(input_dim=X.shape[1], latent_dim=config_obj.LATENT_DIM,
                           shared_dims=config_obj.SHARED_DIMS, tower_dims=config_obj.TOWER_DIMS,
                           dropout_rate=config_obj.DROPOUT_RATE, grl_lambda=config_obj.GRL_LAMBDA)
    rw = config_obj.RANKING_WEIGHT_AFF if task == 'affinity' else config_obj.RANKING_WEIGHT_SPEC
    gw = config_obj.GAP_WEIGHT_AFF if task == 'affinity' else config_obj.GAP_WEIGHT_SPEC
    trainer = MOLMSingleTaskTrainer(model, task=task, aff_pos_weight=aff_pos_weight,
                                     spec_pos_weight=spec_pos_weight, learning_rate=config_obj.LEARNING_RATE,
                                     ranking_weight=rw, gap_weight=gw, focal_gamma=config_obj.FOCAL_GAMMA,
                                     ranking_margin=config_obj.RANKING_MARGIN, gap_margin=config_obj.GAP_MARGIN)
    trainer.fit(X, y_aff, y_spec, epochs=config_obj.EPOCHS, batch_size=config_obj.BATCH_SIZE)
    return model

# ============================================================================
# BASELINE: DeepProjectorDecider NN
# ============================================================================
@tf.keras.utils.register_keras_serializable()
class DeepProjectorDecider(Model):
    def __init__(self, input_dim, intermed_dim=20, proj_dim=1, **kwargs):
        super().__init__(**kwargs)
        self._input_dim = input_dim
        self._intermed_dim = intermed_dim
        self._proj_dim = proj_dim
        self.projector = keras.Sequential([layers.InputLayer(input_shape=(input_dim,)),
                                           layers.Dense(intermed_dim, activation='relu'), layers.Dense(proj_dim)])
        self.decider = keras.Sequential([layers.InputLayer(input_shape=(proj_dim,)), layers.Dense(2, dtype='float32')])
    
    def call(self, x, training=None):
        return self.decider(self.projector(x))
    
    def get_projection(self, x):
        return self.projector(x)
    
    def get_config(self):
        base = super().get_config()
        base.update({'input_dim': self._input_dim, 'intermed_dim': self._intermed_dim, 'proj_dim': self._proj_dim})
        return base

# ============================================================================
# SAVE/LOAD HELPERS
# ============================================================================
def save_phase_data(data, name):
    path = os.path.join(config.OUTPUT_DIR, f"{name}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  ✓ Saved: {path}")

def load_phase_data(name):
    path = os.path.join(config.OUTPUT_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Phase data not found: {path}\n  Run the previous phase first.")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"  ✓ Loaded: {path}")
    return data

def save_results_csv(results_dict, name):
    path = os.path.join(config.OUTPUT_DIR, f"{name}.csv")
    pd.DataFrame([results_dict]).to_csv(path, index=False)
    print(f"  ✓ Saved: {path}")

def save_results_json(results_dict, name):
    path = os.path.join(config.OUTPUT_DIR, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"  ✓ Saved: {path}")

print("✓ Phase 0 loaded: Config, models, utilities ready")
print(f"  OUTPUT_DIR: {config.OUTPUT_DIR}")
print(f"  FEATURE_TYPES: {config.FEATURE_TYPES}")
print(f"  GPU: {len(gpus)} available")
