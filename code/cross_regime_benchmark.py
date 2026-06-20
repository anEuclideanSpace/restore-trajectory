#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-regime phase-classification baseline (the benchmark's headline demonstration).

Harmonized NDVI-trajectory features are computed across the three regimes
(post-fire / afforestation / mining) and a RandomForest baseline is evaluated
both in-regime (5-fold CV) and across regimes (train on A, test on B) to quantify
the cross-regime transfer gap.

Honest framing: phase labels are RS-derived from NDVI, so in-regime accuracy is
partly label-rule re-learning (high by construction). The meaningful signal is the
cross-regime DROP: even the NDVI->phase mapping does not transfer across regimes,
which means the regimes are genuinely different recovery problems (no free
cross-regime generalization). Three well-populated classes are used (relapse is
too rare).
"""
import os, csv, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def fnum(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


CLASSES = ["degraded", "early_restoration", "stable_restoration"]   # 3 well-populated (relapse too rare)


def feats(series):  # series: 1D np array of NDVI ordered in time
    s = np.array([v for v in series if np.isfinite(v)], dtype=float)
    if s.size < 4:
        return None
    x = np.arange(s.size)
    early = s[:max(2, s.size // 5)].mean(); recent = s[-max(2, s.size // 5):].mean()
    slope = float(np.polyfit(x, s, 1)[0]); mn = s.min(); mx = s.max()
    return [s.mean(), early, recent, slope, recent - early, mn, mx, mx - mn, s.std(), recent - mn]


FN = ["ndvi_mean", "ndvi_early", "ndvi_recent", "ndvi_slope", "ndvi_delta",
      "ndvi_min", "ndvi_max", "ndvi_range", "ndvi_std", "ndvi_recovery"]

X = {}; Y = {}
# ---- post-fire ----
ts = {}
for r in load(os.path.join(DATA, "postfire", "RT_postfire_pilot_timeseries.csv")):
    ts.setdefault(r["Event_ID"], []).append((fnum(r["t_rel"]), fnum(r["ndvi"])))
ph = {r["Event_ID"]: r["phase"] for r in load(os.path.join(DATA, "postfire", "RT_postfire_pilot_labels.csv"))}
Xp = []; Yp = []
for eid, seq in ts.items():
    if ph.get(eid) not in CLASSES:
        continue
    seq = [v for _, v in sorted(seq)]; f = feats(np.array(seq))
    if f:
        Xp.append(f); Yp.append(ph[eid])
X["postfire"] = np.array(Xp); Y["postfire"] = np.array(Yp)


# ---- afforestation & mining (NDVI wide 2000-2024) ----
def load_wide(path, phasepath, phasecol):
    wide = {r["unit_id"]: r for r in load(path)}
    lab = {r["unit_id"]: r[phasecol] for r in load(phasepath)}
    yrs = [c for c in next(iter(wide.values())) if c.startswith("ndvi_")]
    Xa = []; Ya = []
    for u, r in wide.items():
        if lab.get(u) not in CLASSES:
            continue
        f = feats(np.array([fnum(r[c]) for c in yrs]))
        if f:
            Xa.append(f); Ya.append(lab[u])
    return np.array(Xa), np.array(Ya)


X["afforestation"], Y["afforestation"] = load_wide(
    os.path.join(DATA, "afforestation_saihanba", "RT_afforestation_saihanba_NDVI_wide_2000_2024.csv"),
    os.path.join(DATA, "afforestation_saihanba", "RT_afforestation_saihanba_phase_v0.csv"), "phase_v0")
X["mining"], Y["mining"] = load_wide(
    os.path.join(DATA, "mining_rehab", "RT_mining_rehab_OPT_wide.csv"),
    os.path.join(DATA, "mining_rehab", "RT_mining_rehab_phase_v0.csv"), "phase_v0")
regs = ["postfire", "afforestation", "mining"]
for g in regs:
    print(f"{g}: n={len(Y[g])}  classes={dict(zip(*np.unique(Y[g], return_counts=True)))}")


def rf():
    return RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=0, n_jobs=-1)


# ---- in-regime 5-fold CV ----
inreg = {}
for g in regs:
    Xi, yi = X[g], Y[g]; skf = StratifiedKFold(5, shuffle=True, random_state=0); sc = []
    for tr, te in skf.split(Xi, yi):
        m = rf().fit(Xi[tr], yi[tr]); sc.append(balanced_accuracy_score(yi[te], m.predict(Xi[te])))
    inreg[g] = float(np.mean(sc)); print(f"in-regime {g}: balacc={inreg[g]:.3f}")
# ---- cross-regime transfer matrix ----
M = np.zeros((3, 3))
for i, a in enumerate(regs):
    m = rf().fit(X[a], Y[a])
    for j, b in enumerate(regs):
        M[i, j] = balanced_accuracy_score(Y[b], m.predict(X[b])) if a != b else inreg[b]
cross = [M[i, j] for i in range(3) for j in range(3) if i != j]
res = dict(
    classes=CLASSES, n=dict((g, int(len(Y[g]))) for g in regs), features=FN,
    in_regime_balacc={g: round(inreg[g], 3) for g in regs},
    transfer_matrix={regs[i]: {regs[j]: round(float(M[i, j]), 3) for j in range(3)} for i in range(3)},
    mean_in_regime=round(float(np.mean(list(inreg.values()))), 3),
    mean_cross_regime=round(float(np.mean(cross)), 3),
    cross_regime_drop=round(float(np.mean(list(inreg.values())) - np.mean(cross)), 3),
    chance_balacc=round(1 / 3, 3),
    interpretation=("In-regime high = NDVI->phase rule re-learnt (labels are RS-derived). "
                    "Cross-regime balacc drops toward chance => the recovery->phase mapping does NOT "
                    "transfer across regimes; each regime is a distinct shift. This is the benchmark's "
                    "core demonstration (and motivates cross-regime as the hard generalization axis)."))
json.dump(res, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: res[k] for k in
                  ["in_regime_balacc", "mean_in_regime", "mean_cross_regime", "cross_regime_drop"]},
                 ensure_ascii=False, indent=1))
# ---- figure: transfer matrix ----
fig, ax = plt.subplots(figsize=(6.2, 5.4))
im = ax.imshow(M, cmap="RdYlGn", vmin=0.33, vmax=1.0)
ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(regs, rotation=20); ax.set_yticklabels(regs)
ax.set_xlabel("test regime"); ax.set_ylabel("train regime")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=12,
                fontweight="bold" if i == j else "normal")
ax.set_title("Cross-regime phase-transfer balanced accuracy\n"
             "diagonal = in-regime CV; off-diagonal = cross-regime transfer (drops toward chance 0.33)")
fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
fig.savefig(os.path.join(HERE, "cross_regime_transfer_matrix.png"), dpi=130, bbox_inches="tight")
print("wrote results.json + cross_regime_transfer_matrix.png")
