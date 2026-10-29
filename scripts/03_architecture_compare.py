# -*- coding: utf-8 -*-
# Part of the Ori-Finder-Pombe reproduction package.
# Run from anywhere; all paths resolve against the repository root.
"""Re-run the encoding x classifier comparison on the 387-benchmark labels.

Mirrors compare_architectures.py (6 encodings x 3 models, stratified 5-fold,
seed 42) but uses the new labelling: candidates overlapping only a single-source
locus are dropped, and "high-confidence" is redefined as a locus supported by
all three evidence lines.
"""

import csv
import sys
import time
from collections import defaultdict

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

sys.stdout.reconfigure(encoding="utf-8")

import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW = str(ROOT_DIR / "data" / "raw")
RES = str(ROOT_DIR / "results")
CANDS = str(ROOT_DIR / "pipeline" / "out")
os.makedirs(RES, exist_ok=True)
CAND = os.path.join(CANDS, "candidates.fasta")
LOCI = os.path.join(RES, "benchmark_v2_loci.tsv")
OUT = os.path.join(RES, "arch_compare")

TRANS = {c: i for i, c in enumerate("ACGT")}
KEY = {"chr1": "I", "chr2": "II", "chr3": "III",
       "chrI": "I", "chrII": "II", "chrIII": "III"}


def seq_to_codes(seq, k):
    arr = np.array([TRANS.get(ch, -1) for ch in seq], dtype=np.int8)
    if k > len(arr):
        return None
    w = sliding_window_view(arr, k)
    valid = (w != -1).all(axis=1)
    if not valid.any():
        return None
    return w[valid].astype(np.uint8) @ (4 ** np.arange(k - 1, -1, -1))


def kmer_vector(seq, k):
    codes = seq_to_codes(seq, k)
    vec = np.zeros(4 ** k, dtype=np.float64)
    if codes is None:
        return vec
    np.add.at(vec, codes, 1)
    s = vec.sum()
    return vec / s if s else vec


def zcurve_vector(seq, k):
    codes = seq_to_codes(seq, k)
    n_prefix = 4 ** (k - 1)
    if codes is None:
        return np.zeros(3 * n_prefix)
    prefix = codes >> 2
    last = codes & 3
    total = len(codes)
    counts = np.bincount(prefix * 4 + last, minlength=n_prefix * 4).reshape(n_prefix, 4)
    nA, nC, nG, nT = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    x = (nA + nG - nC - nT) / total
    y = (nA + nC - nG - nT) / total
    z = (nA + nT - nG - nC) / total
    return np.concatenate([x, y, z])


# ---------- benchmark loci ----------
loci = []
for r in csv.DictReader(open(LOCI, encoding="utf-8"), delimiter="\t"):
    if r["class"] != "benchmark":
        continue
    three = (r["original_source"] == "both" and r["puseq_support"] == "yes")
    loci.append((KEY.get(r["chrom"], r["chrom"]), int(r["start"]), int(r["end"]),
                 2 if three else 1))
by_chrom = defaultdict(list)
for c, s, e, lv in loci:
    by_chrom[c].append((s, e, lv))
print("benchmark loci:", len(loci), "three-source:", sum(1 for x in loci if x[3] == 2))


def label_of(chrom, s, e):
    best = 0
    for bs, be, lv in by_chrom.get(chrom, []):
        ov = min(e, be) - max(s, bs)
        if ov > 0 and (ov / (be - bs) >= 0.5 or ov / (e - s) >= 0.7):
            best = max(best, lv)
    return best


# ---------- candidates (fuzzy already removed by label 0 + explicit list) ----------
FUZZY = set()
for r in csv.DictReader(open(os.path.join(RES, "candidate_labels.tsv"), encoding="utf-8"),
                        delimiter="\t"):
    if r["class"] == "fuzzy":
        FUZZY.add(r["header"])

import re
pat = re.compile(r"([IV]+|chr\w+):(\d+)-(\d+)")
seqs, labels = [], []
name, buf = None, []
recs = []
for line in open(CAND, encoding="utf-8"):
    line = line.rstrip("\n")
    if line.startswith(">"):
        if name is not None:
            recs.append((name, "".join(buf)))
        name, buf = line[1:].strip(), []
    else:
        buf.append(line.strip())
if name is not None:
    recs.append((name, "".join(buf)))

import re as _re
COMP = str.maketrans("ACGTacgt", "TGCAtgca")
for h, seq in recs:
    if h in FUZZY:
        continue
    m = pat.search(h)
    if m is None:
        continue
    chrom = KEY.get(m.group(1), m.group(1))
    s, e = int(m.group(2)), int(m.group(3))
    sm = _re.search(r"\(([+-])\)", h)
    strand = sm.group(1) if sm else "+"
    if strand == "-":
        seq = seq.translate(COMP)[::-1]
    seqs.append(seq.upper())
    labels.append(label_of(chrom, s, e))

y = np.array(labels)
y_bin = (y > 0).astype(int)
print(f"samples={len(y)} neg={int((y == 0).sum())} pos={int((y == 1).sum())} "
      f"three-source={int((y == 2).sum())}")

# ---------- encode ----------
encodings = {}
for k in (4, 5, 6):
    t0 = time.time()
    encodings[f"kmer{k}"] = np.vstack([kmer_vector(s, k) for s in seqs])
    print(f"kmer{k} {encodings[f'kmer{k}'].shape} in {time.time()-t0:.0f}s", flush=True)
for k in (4, 5, 6):
    t0 = time.time()
    encodings[f"zcurve{k}"] = np.vstack([zcurve_vector(s, k) for s in seqs])
    print(f"zcurve{k} {encodings[f'zcurve{k}'].shape} in {time.time()-t0:.0f}s", flush=True)

models = {
    "RBF-SVM": lambda: SVC(kernel="rbf", class_weight="balanced"),
    "RF": lambda: RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                         n_jobs=1, random_state=42),
}
if HAS_LGBM:
    models["LightGBM"] = lambda: LGBMClassifier(n_estimators=500, class_weight="balanced",
                                                random_state=42, verbosity=-1, n_jobs=1)
print("models:", list(models))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(len(y)), y_bin))

rows = []
for enc_name, X in encodings.items():
    for model_name, make in models.items():
        t0 = time.time()
        for fold, (tr, te) in enumerate(folds):
            clf = make()
            clf.fit(X[tr], y_bin[tr])
            pred = clf.predict(X[te])
            score = (clf.decision_function(X[te]) if hasattr(clf, "decision_function")
                     else clf.predict_proba(X[te])[:, 1])
            yt = y_bin[te]
            hc = y[te] == 2
            rows.append({
                "encoding": enc_name, "model": model_name, "fold": fold,
                "ACC": accuracy_score(yt, pred),
                "Sn": recall_score(yt, pred),
                "Prec": precision_score(yt, pred),
                "F1": f1_score(yt, pred),
                "MCC": matthews_corrcoef(yt, pred),
                "AUC": roc_auc_score(yt, score),
                "HC_recall": recall_score(np.ones(hc.sum()), pred[hc]) if hc.sum() else np.nan,
            })
        print(f"{enc_name} x {model_name} done in {time.time()-t0:.0f}s", flush=True)

import os
os.makedirs(OUT, exist_ok=True)
cols = ["encoding", "model", "fold", "ACC", "Sn", "Prec", "F1", "MCC", "AUC", "HC_recall"]
with open(os.path.join(OUT, "cv_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

metrics = ["ACC", "Sn", "Prec", "F1", "MCC", "AUC", "HC_recall"]
with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["encoding", "model"] + [f"{m}_mean" for m in metrics] + [f"{m}_std" for m in metrics])
    for enc_name in encodings:
        for model_name in models:
            sub = [r for r in rows if r["encoding"] == enc_name and r["model"] == model_name]
            means = [f"{np.nanmean([r[m] for r in sub]):.4f}" for m in metrics]
            stds = [f"{np.nanstd([r[m] for r in sub]):.4f}" for m in metrics]
            w.writerow([enc_name, model_name] + means + stds)
print("saved to", OUT)

