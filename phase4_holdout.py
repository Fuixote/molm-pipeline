"""
MOLM Pipeline — Phase 4: Held-Out Mutation Site Evaluation
===========================================================
All models × all features × 8 CDR sites × 2 modes (top/wildtype).

Requires: features.pkl (Phase 1)
Saves: holdout.pkl (per-site results + McNemar)
Runtime: ~90 min
"""
# Auto-import: works when phases are run as local scripts
try:
    config  # Already loaded if Phase 0 ran in this kernel
except NameError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd())
    from phase0_config import *

def predict_model(model, X):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32, device=device))

def tensor_to_numpy(x):
    return x.detach().cpu().numpy()

def held_out_site_evaluation(features, mode='top', epochs=None, latent_dim=None,
                              nn_epochs=50, nn_intermed_dim=20):
    if epochs is None: epochs = config.EPOCHS
    if latent_dim is None: latent_dim = config.LATENT_DIM
    
    residues = config.TOP_RESIDUES if mode == 'top' else config.WILDTYPE_RESIDUES
    mode_name = "TOP-PERFORMING" if mode == 'top' else "WILD-TYPE"
    
    primary_feat, primary_label = get_primary_feat(features)
    X_primary = features['emi'][primary_feat][0]
    X_onehot = features['emi']['onehot'][0]
    X_esm2 = features['emi']['esm2'][0] if 'esm2' in features['emi'] else None
    y_aff = features['emi'][primary_feat][1]
    y_spec = features['emi'][primary_feat][2]
    emi_binding = features['emi']['binding']
    aff_pos_weight = features['emi']['aff_pos_weight']
    spec_pos_weight = features['emi']['spec_pos_weight']
    
    print(f"\n🧬 Held-out evaluation ({mode_name})...")
    print(f"  Sites: {config.MUTATION_SITES_KABAT}, Primary: {primary_label}")
    
    results = {'mode': mode, 'sites': [], 'site_names': [], 'train_sizes': [], 'test_sizes': []}
    
    for i, (site_idx, kabat_pos, residue) in enumerate(zip(
            config.MUTATION_SITES, config.MUTATION_SITES_KABAT, residues)):
        train_idx, test_idx = create_holdout_indices(emi_binding.index, site_idx, residue, mode)
        if len(test_idx) < 10:
            print(f"  ⚠ Site {kabat_pos}: only {len(test_idx)} test samples, skipping")
            continue
        
        ya_tr, ya_te = y_aff[train_idx], y_aff[test_idx]
        ys_tr, ys_te = y_spec[train_idx], y_spec[test_idx]
        print(f"\n  Site Kabat {kabat_pos} (res '{residue}'): Train {len(train_idx)}, Test {len(test_idx)}")
        
        # ========== MOLM + MOLM-ST on ALL features ==========
        molm_preds = {}  # {feat_type: (pred_aff, pred_spec)}
        st_preds = {}    # {feat_type: (pred_aff, pred_spec)}
        pred_aff_primary = pred_spec_primary = None
        st_pred_aff_primary = st_pred_spec_primary = None
        
        for fi, feat_type in enumerate(config.FEATURE_TYPES):
            if feat_type not in features['emi']: continue
            fl = FEAT_LABELS.get(feat_type, feat_type)
            X_feat = features['emi'][feat_type][0]
            
            # MOLM
            hard_reset_rng(SEED + i + fi * 100, f"MOLM-{fl} site {kabat_pos}")
            model = DiagnosticMOLM(input_dim=X_feat.shape[1], latent_dim=latent_dim,
                                   shared_dims=config.SHARED_DIMS, tower_dims=config.TOWER_DIMS,
                                   dropout_rate=config.DROPOUT_RATE, grl_lambda=config.GRL_LAMBDA)
            trainer = DiagnosticTrainer(model, aff_pos_weight=aff_pos_weight,
                                        spec_pos_weight=spec_pos_weight, learning_rate=config.LEARNING_RATE)
            trainer.fit(X_feat[train_idx], ya_tr, ys_tr, epochs=epochs, batch_size=config.BATCH_SIZE, verbose=0)
            out = predict_model(model, X_feat[test_idx])
            aff_score = tensor_to_numpy(out['logit_aff'])
            spec_score = tensor_to_numpy(out['logit_spec'])
            aff_prob = tensor_to_numpy(out['aff_prob'])
            spec_prob = tensor_to_numpy(out['spec_prob'])
            m_pa = (aff_score > 0).astype(np.int64)
            m_ps = (spec_score > 0).astype(np.int64)
            del model, trainer; gc.collect(); torch.cuda.empty_cache()
            
            mk = f'molm_{feat_type}'
            for suffix, val in [('_affinity_accs', (m_pa==ya_te).mean()), ('_specificity_accs', (m_ps==ys_te).mean()),
                                ('_aff_aucs', safe_auc(ya_te, aff_prob)),
                                ('_spec_aucs', safe_auc(ys_te, spec_prob)),
                                ('_aff_mccs', safe_mcc(ya_te, m_pa)), ('_spec_mccs', safe_mcc(ys_te, m_ps))]:
                results.setdefault(mk+suffix, []).append(val)
            
            if feat_type == primary_feat:
                pred_aff_primary, pred_spec_primary = m_pa, m_ps
            molm_preds[feat_type] = (m_pa, m_ps)
            
            print(f"    MOLM ({fl:12s}) Aff:{(m_pa==ya_te).mean():.4f} MCC:{safe_mcc(ya_te,m_pa):.3f} | Spec:{(m_ps==ys_te).mean():.4f} MCC:{safe_mcc(ys_te,m_ps):.3f}")
            
            # MOLM-ST
            if config.RUN_MOLM_ST:
                st_a = train_molm_st(X_feat[train_idx], ya_tr, ys_tr, 'affinity', config,
                                     aff_pos_weight, spec_pos_weight, i+500+fi*100)
                out_a = predict_model(st_a, X_feat[test_idx])
                st_pa = (tensor_to_numpy(out_a['logit_aff']) > 0).astype(np.int64)
                del st_a; gc.collect(); torch.cuda.empty_cache()
                
                st_s = train_molm_st(X_feat[train_idx], ya_tr, ys_tr, 'specificity', config,
                                     aff_pos_weight, spec_pos_weight, i+600+fi*100)
                out_s = predict_model(st_s, X_feat[test_idx])
                st_ps = (tensor_to_numpy(out_s['logit_spec']) > 0).astype(np.int64)
                del st_s; gc.collect(); torch.cuda.empty_cache()
                
                sk = f'molm_st_{feat_type}'
                for suffix, val in [('_affinity_accs', (st_pa==ya_te).mean()), ('_specificity_accs', (st_ps==ys_te).mean()),
                                    ('_aff_mccs', safe_mcc(ya_te, st_pa)), ('_spec_mccs', safe_mcc(ys_te, st_ps))]:
                    results.setdefault(sk+suffix, []).append(val)
                
                if feat_type == primary_feat:
                    st_pred_aff_primary, st_pred_spec_primary = st_pa, st_ps
                st_preds[feat_type] = (st_pa, st_ps)
                
                print(f"    MOLM-ST ({fl:12s}) Aff:{(st_pa==ya_te).mean():.4f} MCC:{safe_mcc(ya_te,st_pa):.3f} | Spec:{(st_ps==ys_te).mean():.4f} MCC:{safe_mcc(ys_te,st_ps):.3f}")
        
        # ========== NN on all features ==========
        nn_preds = {}  # {feat_type: (pred_aff, pred_spec)}
        for nn_ft, nn_X, seed_base in [('onehot', X_onehot, 0), ('esm2', X_esm2, 300), ('fusion_esm2', X_primary, 200)]:
            if nn_X is None: continue
            fl = FEAT_LABELS.get(nn_ft, nn_ft)
            
            hard_reset_rng(SEED + i*100 + seed_base, f"NN-{fl} site {kabat_pos} aff")
            nn_a = train_deep_projector_decider(
                nn_X[train_idx], ya_tr, input_dim=nn_X.shape[1], intermed_dim=nn_intermed_dim,
                epochs=nn_epochs, batch_size=50, seed=SEED + i*100 + seed_base)
            nn_logits_a = predict_deep_projector_logits(nn_a, nn_X[test_idx])
            nn_pa = np.argmax(nn_logits_a, 1); del nn_a; gc.collect(); torch.cuda.empty_cache()
            
            hard_reset_rng(SEED + i*100 + seed_base + 50, f"NN-{fl} site {kabat_pos} spec")
            nn_s = train_deep_projector_decider(
                nn_X[train_idx], ys_tr, input_dim=nn_X.shape[1], intermed_dim=nn_intermed_dim,
                epochs=nn_epochs, batch_size=50, seed=SEED + i*100 + seed_base + 50)
            nn_logits_s = predict_deep_projector_logits(nn_s, nn_X[test_idx])
            nn_ps = np.argmax(nn_logits_s, 1); del nn_s; gc.collect(); torch.cuda.empty_cache()
            
            nn_preds[nn_ft] = (nn_pa, nn_ps)
            nk = f'nn_{nn_ft}'
            for suffix, val in [('_affinity_accs', (nn_pa==ya_te).mean()), ('_specificity_accs', (nn_ps==ys_te).mean()),
                                ('_aff_mccs', safe_mcc(ya_te, nn_pa)), ('_spec_mccs', safe_mcc(ys_te, nn_ps))]:
                results.setdefault(nk+suffix, []).append(val)
            
            print(f"    NN ({fl:12s})    Aff:{(nn_pa==ya_te).mean():.4f} MCC:{safe_mcc(ya_te,nn_pa):.3f} | Spec:{(nn_ps==ys_te).mean():.4f} MCC:{safe_mcc(ys_te,nn_ps):.3f}")
        gc.collect()
        
        # ========== LDA on ALL features ==========
        lda_preds = {}  # {feat_type: (pred_aff, pred_spec)}
        for lda_ft in config.FEATURE_TYPES:
            if lda_ft not in features['emi']: continue
            fl = FEAT_LABELS.get(lda_ft, lda_ft)
            X_lda = features['emi'][lda_ft][0]
            
            lda_a = LDA(); lda_a.fit(X_lda[train_idx], ya_tr); lda_pa = lda_a.predict(X_lda[test_idx])
            lda_s = LDA(); lda_s.fit(X_lda[train_idx], ys_tr); lda_ps = lda_s.predict(X_lda[test_idx])
            lda_preds[lda_ft] = (lda_pa, lda_ps)
            
            lk = f'lda_{lda_ft}'
            for suffix, val in [('_affinity_accs', (lda_pa==ya_te).mean()), ('_specificity_accs', (lda_ps==ys_te).mean()),
                                ('_aff_mccs', safe_mcc(ya_te, lda_pa)), ('_spec_mccs', safe_mcc(ys_te, lda_ps))]:
                results.setdefault(lk+suffix, []).append(val)
            print(f"    LDA ({fl:12s})  Aff:{(lda_pa==ya_te).mean():.4f} MCC:{safe_mcc(ya_te,lda_pa):.3f} | Spec:{(lda_ps==ys_te).mean():.4f} MCC:{safe_mcc(ys_te,lda_ps):.3f}")
        
        # Keep backward-compatible 'lda' keys pointing to OneHot results
        if 'onehot' in lda_preds:
            lda_pa, lda_ps = lda_preds['onehot']
            for suffix, val in [('_affinity_accs', (lda_pa==ya_te).mean()), ('_specificity_accs', (lda_ps==ys_te).mean()),
                                ('_aff_mccs', safe_mcc(ya_te, lda_pa)), ('_spec_mccs', safe_mcc(ys_te, lda_ps))]:
                results.setdefault(f'lda{suffix}', []).append(val)
        
        # ========== Store metadata ==========
        results['sites'].append(site_idx)
        results['site_names'].append(f"Kabat {kabat_pos}")
        results['train_sizes'].append(len(train_idx))
        results['test_sizes'].append(len(test_idx))
        
        # ========== McNemar: MOLM(primary) vs baselines AND MOLM(OneHot) vs baselines ==========
        for mcn_label, mcn_key, molm_pa, molm_ps, st_pa_ref, st_ps_ref in [
            (primary_label, 'mcnemar', pred_aff_primary, pred_spec_primary, st_pred_aff_primary, st_pred_spec_primary),
            ('OneHot', 'mcnemar_oh', molm_preds.get('onehot', (None,None))[0], molm_preds.get('onehot', (None,None))[1],
             st_preds.get('onehot', (None,None))[0], st_preds.get('onehot', (None,None))[1])
        ]:
            if molm_pa is None: continue
            if mcn_key not in results:
                results[mcn_key] = {k: {'aff': [], 'spec': []} for k in
                                    ['vs_molm_st', 'vs_nn_oh', 'vs_nn_esm2', 'vs_nn_fesm2', 'vs_lda']}
            
            if st_pa_ref is not None:
                _, p = mcnemar_test(ya_te, molm_pa, st_pa_ref); results[mcn_key]['vs_molm_st']['aff'].append(p)
                _, p = mcnemar_test(ys_te, molm_ps, st_ps_ref); results[mcn_key]['vs_molm_st']['spec'].append(p)
            
            for vs_key, vs_ft in [('vs_nn_oh', 'onehot'), ('vs_nn_esm2', 'esm2'), ('vs_nn_fesm2', 'fusion_esm2')]:
                if vs_ft in nn_preds:
                    _, p = mcnemar_test(ya_te, molm_pa, nn_preds[vs_ft][0].astype(np.int64)); results[mcn_key][vs_key]['aff'].append(p)
                    _, p = mcnemar_test(ys_te, molm_ps, nn_preds[vs_ft][1].astype(np.int64)); results[mcn_key][vs_key]['spec'].append(p)
            
            if 'onehot' in lda_preds:
                _, p = mcnemar_test(ya_te, molm_pa, lda_preds['onehot'][0].astype(np.int64)); results[mcn_key]['vs_lda']['aff'].append(p)
                _, p = mcnemar_test(ys_te, molm_ps, lda_preds['onehot'][1].astype(np.int64)); results[mcn_key]['vs_lda']['spec'].append(p)
    
    # ========== Summary ==========
    if results['sites']:
        n_sites = len(results['sites'])
        print(f"\n  {'='*90}")
        print(f"  {mode_name} HELD-OUT SUMMARY (mean ± std across {n_sites} sites)")
        print(f"  {'='*90}")
        print(f"  {'Model':<30} {'Aff Acc':>12} {'Aff MCC':>10} {'Spec Acc':>12} {'Spec MCC':>10}")
        print(f"  {'-'*90}")
        
        # Compute mean AND std for all metric keys
        all_keys = [k for k in results if k.endswith('_accs') or k.endswith('_mccs')]
        for k in all_keys:
            if isinstance(results[k], list) and results[k]:
                results[k.replace('accs', 'acc_mean').replace('mccs', 'mcc_mean')] = np.nanmean(results[k])
                results[k.replace('accs', 'acc_std').replace('mccs', 'mcc_std')] = np.nanstd(results[k])
        
        def fmt_acc(mean_k, std_k):
            m = results.get(mean_k, 0)
            s = results.get(std_k, 0)
            return f"{m:.4f}±{s:.4f}" if s > 0 else f"{m:.4f}"
        
        for ft in config.FEATURE_TYPES:
            fl = FEAT_LABELS.get(ft, ft)
            for mp, mn in [('molm', 'MOLM'), ('molm_st', 'MOLM-ST')]:
                ak = f'{mp}_{ft}_affinity_acc_mean'
                if ak in results:
                    print(f"  {mn+' ('+fl+')':<30} "
                          f"{fmt_acc(f'{mp}_{ft}_affinity_acc_mean', f'{mp}_{ft}_affinity_acc_std'):>12} "
                          f"{results.get(f'{mp}_{ft}_aff_mcc_mean',0):>10.3f} "
                          f"{fmt_acc(f'{mp}_{ft}_specificity_acc_mean', f'{mp}_{ft}_specificity_acc_std'):>12} "
                          f"{results.get(f'{mp}_{ft}_spec_mcc_mean',0):>10.3f}")
        
        for nk, nl in [('nn_onehot', 'NN (OneHot)'), ('nn_esm2', 'NN (ESM2)'),
                        ('nn_fusion_esm2', f'NN ({primary_label})'),
                        ('lda_onehot', 'LDA (OneHot)'), ('lda_esm2', 'LDA (ESM2)'),
                        ('lda_fusion_esm2', f'LDA ({primary_label})')]:
            ak = f'{nk}_affinity_acc_mean'
            if ak in results:
                print(f"  {nl:<30} "
                      f"{fmt_acc(f'{nk}_affinity_acc_mean', f'{nk}_affinity_acc_std'):>12} "
                      f"{results.get(f'{nk}_aff_mcc_mean',0):>10.3f} "
                      f"{fmt_acc(f'{nk}_specificity_acc_mean', f'{nk}_specificity_acc_std'):>12} "
                      f"{results.get(f'{nk}_spec_mcc_mean',0):>10.3f}")
        
        # Also keep backward-compatible 'lda' keys
        for bk in ['lda_affinity_acc_mean', 'lda_specificity_acc_mean', 'lda_aff_mcc_mean', 'lda_spec_mcc_mean']:
            if bk not in results and bk.replace('lda_', 'lda_onehot_') in results:
                results[bk] = results[bk.replace('lda_', 'lda_onehot_')]
        
        # Print McNemar for BOTH primary and OneHot
        for mcn_key, mcn_label in [('mcnemar', f'MOLM({primary_label})'), ('mcnemar_oh', 'MOLM(OneHot)')]:
            if mcn_key in results:
                print(f"\n  McNemar ({mcn_label} vs baselines):")
                for vk, vd in results[mcn_key].items():
                    if vd['aff']:
                        ns_a = sum(1 for p in vd['aff'] if p < 0.05)
                        ns_s = sum(1 for p in vd['spec'] if p < 0.05)
                        print(f"    {vk:20s}: Aff {ns_a}/{len(vd['aff'])} sig | Spec {ns_s}/{len(vd['spec'])} sig")
    
    return results

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("PHASE 4: Held-Out Mutation Site Evaluation")
    print("=" * 60)
    
    features = load_phase_data("features")
    
    holdout_top = held_out_site_evaluation(features, mode='top')
    holdout_wt = held_out_site_evaluation(features, mode='wildtype')
    
    save_phase_data({'top': holdout_top, 'wt': holdout_wt}, "holdout")
    
    # CSV export
    for mode_key, res in [('top', holdout_top), ('wt', holdout_wt)]:
        rows = []
        for si, sn in enumerate(res.get('site_names', [])):
            row = {'site': sn, 'train': res['train_sizes'][si], 'test': res['test_sizes'][si]}
            for k, v in res.items():
                if isinstance(v, list) and len(v) > si and k not in ['sites', 'site_names', 'train_sizes', 'test_sizes']:
                    row[k] = v[si]
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(config.OUTPUT_DIR, f"holdout_{mode_key}.csv"), index=False)
    
    print(f"\n✓ Phase 4 complete in {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")
