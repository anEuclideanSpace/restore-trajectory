# -*- coding: utf-8 -*-
"""
Label cards and leakage audit for the afforestation and mining regimes.

For each regime we report how well a standard RandomForest probe trained on raw
NDVI trajectories can recover the phase label. When the label is itself derived
from NDVI (Tier-B) the probe accuracy is high by construction (leakage). When an
independent structure source is available (CLCD forest fraction, Tier-A) the same
probe is evaluated against that independent label, and the agreement between the
two label sources is measured with Cohen's kappa.

A standard RF baseline is used throughout (no physics-constrained method here).
"""
import os
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score as BA, cohen_kappa_score as KAP
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
LAB = {'degraded': 0, 'early_restoration': 1, 'stable_restoration': 2}


def rawndvi(df):
    cols = [f'ndvi_{y}' for y in range(2000, 2024) if f'ndvi_{y}' in df.columns]
    M = df[cols].values.astype(float); rm = np.nanmean(M, 1, keepdims=True)
    return np.nan_to_num(np.where(np.isnan(M), rm, M))


def rf():
    return RandomForestClassifier(200, random_state=0, class_weight='balanced', n_jobs=-1)


def probe_BA(X, y, seed=0):
    tr, te = train_test_split(np.arange(len(y)), test_size=.3, random_state=seed, stratify=y)
    return BA(y[te], rf().fit(X[tr], y[tr]).predict(X[te]))


# ---------- load ----------
AFF = os.path.join(DATA, "afforestation_saihanba")
MINE = os.path.join(DATA, "mining_rehab")
an = pd.read_csv(os.path.join(AFF, "RT_afforestation_saihanba_NDVI_wide_2000_2024.csv"))
ap = pd.read_csv(os.path.join(AFF, "RT_afforestation_saihanba_phase_v0.csv"))
cs = pd.read_csv(os.path.join(AFF, "RT_afforestation_saihanba_CLCD_unit_summary.csv"))[['unit_id', 'clcd_forest_gain']]
A = an.merge(ap, on='unit_id').merge(cs, on='unit_id'); A = A[A.phase_v0.isin(LAB)].reset_index(drop=True)
mn = pd.read_csv(os.path.join(MINE, "RT_mining_rehab_OPT_wide.csv"))
mp = pd.read_csv(os.path.join(MINE, "RT_mining_rehab_phase_v0.csv"))
Mi = mn.merge(mp, on='unit_id'); Mi = Mi[Mi.phase_v0.isin(LAB)].reset_index(drop=True)

# ---------- labels ----------
ff = A['clcd_ff_final'].values; gain = A['clcd_forest_gain'].values
yA_clcd = np.where(ff >= 0.6, 2, np.where((gain > 0.15) | (ff >= 0.30), 1, 0))   # INDEPENDENT (Tier-A)
yA_ndvi = A.phase_v0.map(LAB).values                                              # leaky (Tier-B)
yM_ndvi = Mi.phase_v0.map(LAB).values                                            # leaky
XA = rawndvi(A); XM = rawndvi(Mi)
print("\n================ LABEL CARDS + LEAKAGE AUDIT ================")
print(f"[AFFOREST] n={len(A)}")
print(f"  NDVI-probe BA -> NDVI-label (Tier-B, leaky) = {probe_BA(XA, yA_ndvi):.3f}")
print(f"  NDVI-probe BA -> CLCD-label (Tier-A, indep) = {probe_BA(XA, yA_clcd):.3f}")
print(f"  cross-source kappa(NDVI-label, CLCD-label) = {KAP(yA_ndvi, yA_clcd):.3f}  -> Tier-A label = CLCD")
print(f"[MINING] n={len(Mi)}")
print(f"  NDVI-probe BA -> NDVI-label (Tier-B, leaky) = {probe_BA(XM, yM_ndvi):.3f}")

# ---------- MINING external-status cross-check ----------
ext = Mi['aml_status_ext'].fillna('no_aml_match')
mask = ext.isin(['reclamation_complete', 'aml_listed_not_complete']).values
print(f"\n================ MINING EXTERNAL-STATUS CROSS-CHECK ================")
print(f"  external AML coverage: {mask.sum()}/{len(Mi)}  "
      f"(complete={int((ext == 'reclamation_complete').sum())}, "
      f"listed-not-complete={int((ext == 'aml_listed_not_complete').sum())})")
if mask.sum() > 20:
    # external status -> expected coarse state: complete ~ recovered, listed-not-complete ~ not yet
    ext_state = np.where(ext[mask].values == 'reclamation_complete', 1, 0)   # 1 = reclaimed-evidence, 0 = not
    ndvi_state = (yM_ndvi[mask] >= 2).astype(int)                            # NDVI says stable?
    print(f"  agreement(NDVI 'stable' vs external 'reclamation_complete') kappa={KAP(ext_state, ndvi_state):.3f}")
    print(f"    note: external coverage is small ({mask.sum()}), complete only "
          f"{(ext == 'reclamation_complete').sum()} -> status label is sparse")
