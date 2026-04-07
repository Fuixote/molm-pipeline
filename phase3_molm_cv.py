"""
MOLM Pipeline — Phase 3: MOLM + MOLM-ST Cross-Validation
=========================================================
Full feature grid: MOLM and MOLM-ST on OneHot, ESM2, Fusion-ESM2.

Requires: features.pkl (from Phase 1)
Saves: molm_cv.pkl (CV results for all feature types)
Runtime: ~2 hours (3 features × MOLM + MOLM-ST × 5 folds × 25 epochs)
"""
# Auto-import: works both as .py file AND pasted into notebook cells
try:
    config  # Already loaded if Phase 0 ran in this kernel
except NameError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "/kaggle/working")
    from phase0_config import *

def export_epoch_diagnostics_csv(trainer, save_path):
    rows = [{'epoch': d.epoch+1, 'loss_total': d.loss_total,
             'loss_cls_aff': d.loss_cls_aff, 'loss_cls_spec': d.loss_cls_spec,
             'acc_aff': d.acc_aff, 'acc_spec': d.acc_spec,
             'aff_gap': d.aff_gap, 'spec_gap': d.spec_gap,
             'grad_cosine_aff_spec': d.grad_cosine_aff_spec,
             'iso_aff_rho': d.iso_aff_rho, 'iso_spec_rho': d.iso_spec_rho,
             'igg_aff_rho': d.igg_aff_rho, 'igg_spec_rho': d.igg_spec_rho,
            } for d in trainer.epoch_diagnostics]
    pd.DataFrame(rows).to_csv(save_path, index=False)

def plot_paper_diagnostics(trainer, save_path):
    diags = trainer.epoch_diagnostics
    epochs = np.array([d.epoch+1 for d in diags])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    grad_cos = np.array([d.grad_cosine_aff_spec for d in diags])
    axes[0].plot(epochs, grad_cos, color='purple', lw=2)
    axes[0].axhline(0, ls="--", color='red', alpha=0.5)
    axes[0].fill_between(epochs, grad_cos, 0, where=grad_cos<0, alpha=0.3, color='red', label='Conflict zone')
    axes[0].set_title("(A) Grad Cosine (Aff vs Spec)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Epoch"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, [d.aff_gap for d in diags], label="Affinity gap", color='#3498db', lw=2)
    axes[1].plot(epochs, [d.spec_gap for d in diags], label="Specificity gap", color='#e74c3c', lw=2)
    axes[1].axhline(0, ls="--", color='black', alpha=0.5)
    axes[1].set_title("(B) Score Gaps (pos - neg)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Epoch"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    
    gen_e = [d.epoch+1 for d in diags if d.iso_aff_rho is not None]
    if gen_e:
        axes[2].plot(gen_e, [d.iso_aff_rho for d in diags if d.iso_aff_rho is not None], 'o-', label='ISO aff', color='#3498db')
        axes[2].plot(gen_e, [d.iso_spec_rho for d in diags if d.iso_spec_rho is not None], 'o-', label='ISO spec', color='#e74c3c')
        axes[2].plot(gen_e, [d.igg_aff_rho for d in diags if d.igg_aff_rho is not None], 's--', label='IgG aff', color='#1a5276')
        axes[2].plot(gen_e, [d.igg_spec_rho for d in diags if d.igg_spec_rho is not None], 's--', label='IgG spec', color='#922b21')
        axes[2].legend(fontsize=8)
    else:
        axes[2].text(0.5, 0.5, "No generalization eval points", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set_title("(C) Generalization (Spearman ρ)", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Epoch"); axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.show(); plt.close()

def kfold_cv_molm(X, y_aff, y_spec, feat_label='', n_folds=5,
                  aff_pos_weight=1.0, spec_pos_weight=1.0,
                  X_iso=None, y_iso_aff=None, y_iso_spec=None,
                  X_igg=None, y_igg_aff=None, y_igg_spec=None):
    print(f"\n🔄 MOLM CV [{feat_label}] ({n_folds}-fold)...")
    skf, y_joint = get_stratified_kfold(y_aff, y_spec, n_splits=n_folds, random_state=SEED)
    
    results = {'affinity_accs': [], 'specificity_accs': [],
               'aff_aucs': [], 'spec_aucs': [], 'aff_aps': [], 'spec_aps': [],
               'aff_mccs': [], 'spec_mccs': []}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_joint)):
        print(f"  Fold {fold+1}/{n_folds}:")
        hard_reset_rng(SEED + fold, f"MOLM CV {feat_label} fold {fold}")
        
        model = DiagnosticMOLM(input_dim=X.shape[1], latent_dim=config.LATENT_DIM,
                               shared_dims=config.SHARED_DIMS, tower_dims=config.TOWER_DIMS,
                               dropout_rate=config.DROPOUT_RATE, grl_lambda=config.GRL_LAMBDA)
        trainer = DiagnosticTrainer(model, aff_pos_weight=aff_pos_weight,
                                    spec_pos_weight=spec_pos_weight, learning_rate=config.LEARNING_RATE)
        trainer.fit(X[train_idx], y_aff[train_idx], y_spec[train_idx],
                    epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                    X_iso=X_iso, y_iso_aff=y_iso_aff, y_iso_spec=y_iso_spec,
                    X_igg=X_igg, y_igg_aff=y_igg_aff, y_igg_spec=y_igg_spec, verbose=0)
        
        # Save diagnostics
        fs = f"_{feat_label}" if feat_label else ""
        if config.SAVE_EPOCH_CSV:
            export_epoch_diagnostics_csv(trainer, os.path.join(config.OUTPUT_DIR, f"epoch_diag_fold_{fold+1}{fs}.csv"))
        if config.SAVE_PAPER_FIG:
            plot_paper_diagnostics(trainer, os.path.join(config.OUTPUT_DIR, f"paper_diag_fold_{fold+1}{fs}.png"))
        
        out = model(tf.constant(X[val_idx], tf.float32), training=False)
        pred_a = (out['aff_score'].numpy() > 0).astype(int)
        pred_s = (out['spec_score'].numpy() > 0).astype(int)
        ya_v, ys_v = y_aff[val_idx], y_spec[val_idx]
        
        results['affinity_accs'].append((pred_a == ya_v).mean())
        results['specificity_accs'].append((pred_s == ys_v).mean())
        results['aff_aucs'].append(safe_auc(ya_v, out['aff_prob'].numpy()))
        results['spec_aucs'].append(safe_auc(ys_v, out['spec_prob'].numpy()))
        results['aff_aps'].append(safe_auc(ya_v, out['aff_prob'].numpy(), 'pr'))
        results['spec_aps'].append(safe_auc(ys_v, out['spec_prob'].numpy(), 'pr'))
        results['aff_mccs'].append(safe_mcc(ya_v, pred_a))
        results['spec_mccs'].append(safe_mcc(ys_v, pred_s))
        
        print(f"    Aff: {results['affinity_accs'][-1]:.4f} AUC:{results['aff_aucs'][-1]:.4f} MCC:{results['aff_mccs'][-1]:.3f}")
        print(f"    Spec: {results['specificity_accs'][-1]:.4f} AUC:{results['spec_aucs'][-1]:.4f} MCC:{results['spec_mccs'][-1]:.3f}")
        del model, trainer; tf.keras.backend.clear_session(); gc.collect()
    
    for k in list(results.keys()):
        results[f'{k}_mean'] = np.nanmean(results[k])
        results[f'{k}_std'] = np.nanstd(results[k])
    results['affinity_mean'] = results['affinity_accs_mean']
    results['affinity_std'] = results['affinity_accs_std']
    results['specificity_mean'] = results['specificity_accs_mean']
    results['specificity_std'] = results['specificity_accs_std']
    
    print(f"  📊 Aff: {results['affinity_mean']:.4f}±{results['affinity_std']:.4f} AUC:{results['aff_aucs_mean']:.4f} MCC:{results['aff_mccs_mean']:.3f}")
    print(f"  📊 Spec: {results['specificity_mean']:.4f}±{results['specificity_std']:.4f} AUC:{results['spec_aucs_mean']:.4f} MCC:{results['spec_mccs_mean']:.3f}")
    return results

def kfold_cv_molm_st(X, y_aff, y_spec, feat_label='', n_folds=5,
                     aff_pos_weight=1.0, spec_pos_weight=1.0):
    print(f"\n🔄 MOLM-ST CV [{feat_label}] ({n_folds}-fold)...")
    skf, y_joint = get_stratified_kfold(y_aff, y_spec, n_splits=n_folds, random_state=SEED)
    aff_accs, spec_accs = [], []
    aff_mccs, spec_mccs = [], []
    aff_aucs, spec_aucs = [], []
    
    for fold, (tr, va) in enumerate(skf.split(X, y_joint)):
        # Affinity model
        m_a = train_molm_st(X[tr], y_aff[tr], y_spec[tr], 'affinity', config, aff_pos_weight, spec_pos_weight, 500+fold)
        out_a = m_a(tf.constant(X[va], tf.float32), training=False)
        pred_a = (out_a['aff_score'].numpy() > 0).astype(int)
        aff_accs.append((pred_a == y_aff[va]).mean())
        aff_mccs.append(safe_mcc(y_aff[va], pred_a))
        aff_aucs.append(safe_auc(y_aff[va], out_a['aff_prob'].numpy()))
        del m_a; tf.keras.backend.clear_session(); gc.collect()
        
        # Specificity model
        m_s = train_molm_st(X[tr], y_aff[tr], y_spec[tr], 'specificity', config, aff_pos_weight, spec_pos_weight, 600+fold)
        out_s = m_s(tf.constant(X[va], tf.float32), training=False)
        pred_s = (out_s['spec_score'].numpy() > 0).astype(int)
        spec_accs.append((pred_s == y_spec[va]).mean())
        spec_mccs.append(safe_mcc(y_spec[va], pred_s))
        spec_aucs.append(safe_auc(y_spec[va], out_s['spec_prob'].numpy()))
        del m_s; tf.keras.backend.clear_session(); gc.collect()
        
        print(f"  Fold {fold+1}: Aff {aff_accs[-1]:.4f} MCC:{aff_mccs[-1]:.3f} AUC:{aff_aucs[-1]:.4f} | Spec {spec_accs[-1]:.4f} MCC:{spec_mccs[-1]:.3f} AUC:{spec_aucs[-1]:.4f}")
    
    res = {'affinity_mean': np.mean(aff_accs), 'affinity_std': np.std(aff_accs),
           'specificity_mean': np.mean(spec_accs), 'specificity_std': np.std(spec_accs),
           'affinity_accs': aff_accs, 'specificity_accs': spec_accs,
           'aff_mccs': aff_mccs, 'aff_mccs_mean': np.nanmean(aff_mccs), 'aff_mccs_std': np.nanstd(aff_mccs),
           'spec_mccs': spec_mccs, 'spec_mccs_mean': np.nanmean(spec_mccs), 'spec_mccs_std': np.nanstd(spec_mccs),
           'aff_aucs': aff_aucs, 'aff_aucs_mean': np.nanmean(aff_aucs), 'aff_aucs_std': np.nanstd(aff_aucs),
           'spec_aucs': spec_aucs, 'spec_aucs_mean': np.nanmean(spec_aucs), 'spec_aucs_std': np.nanstd(spec_aucs)}
    print(f"  📊 MOLM-ST [{feat_label}]: Aff {res['affinity_mean']:.4f}±{res['affinity_std']:.4f} MCC:{res['aff_mccs_mean']:.3f} AUC:{res['aff_aucs_mean']:.4f}")
    print(f"                           Spec {res['specificity_mean']:.4f}±{res['specificity_std']:.4f} MCC:{res['spec_mccs_mean']:.3f} AUC:{res['spec_aucs_mean']:.4f}")
    return res

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("PHASE 3: MOLM + MOLM-ST Cross-Validation (Full Grid)")
    print("=" * 60)
    
    features = load_phase_data("features")
    aff_pw = features['emi']['aff_pos_weight']
    spec_pw = features['emi']['spec_pos_weight']
    
    all_molm_cv = {}
    all_molm_st_cv = {}
    
    for feat_type in config.FEATURE_TYPES:
        if feat_type not in features['emi']: continue
        fl = FEAT_LABELS.get(feat_type, feat_type)
        X, y_aff, y_spec = features['emi'][feat_type]
        
        # Get ISO/IgG gen data for this feature type
        X_iso = features['iso'].get(feat_type, (None,))[0] if feat_type in features['iso'] else None
        y_iso_aff = features['iso'][feat_type][1] if X_iso is not None else None
        y_iso_spec = features['iso'][feat_type][2] if X_iso is not None else None
        X_igg = features['igg'].get(feat_type, (None,))[0] if feat_type in features['igg'] else None
        y_igg_aff = features['igg'][feat_type][1] if X_igg is not None else None
        y_igg_spec = features['igg'][feat_type][2] if X_igg is not None else None
        
        all_molm_cv[feat_type] = kfold_cv_molm(
            X, y_aff, y_spec, feat_label=fl, n_folds=config.CV_FOLDS,
            aff_pos_weight=aff_pw, spec_pos_weight=spec_pw,
            X_iso=X_iso, y_iso_aff=y_iso_aff, y_iso_spec=y_iso_spec,
            X_igg=X_igg, y_igg_aff=y_igg_aff, y_igg_spec=y_igg_spec)
        
        if config.RUN_MOLM_ST:
            all_molm_st_cv[feat_type] = kfold_cv_molm_st(
                X, y_aff, y_spec, feat_label=fl, n_folds=config.CV_FOLDS,
                aff_pos_weight=aff_pw, spec_pos_weight=spec_pw)
    
    save_phase_data({'molm_cv': all_molm_cv, 'molm_st_cv': all_molm_st_cv}, "molm_cv")
    
    # Print grid
    print(f"\n{'='*65}")
    print("CV GRID — MOLM + MOLM-ST")
    print(f"{'='*65}")
    for ft in config.FEATURE_TYPES:
        fl = FEAT_LABELS.get(ft, ft)
        if ft in all_molm_cv:
            r = all_molm_cv[ft]
            print(f"  MOLM ({fl:12s}): Aff {r['affinity_mean']:.4f}±{r['affinity_std']:.4f}, Spec {r['specificity_mean']:.4f}±{r['specificity_std']:.4f}")
        if ft in all_molm_st_cv:
            r = all_molm_st_cv[ft]
            print(f"  MOLM-ST ({fl:12s}): Aff {r['affinity_mean']:.4f}±{r['affinity_std']:.4f}, Spec {r['specificity_mean']:.4f}±{r['specificity_std']:.4f}")
    
    print(f"\n✓ Phase 3 complete in {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")
