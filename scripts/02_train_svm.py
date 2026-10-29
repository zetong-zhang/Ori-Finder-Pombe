# -*- coding: utf-8 -*-
# Part of the Ori-Finder-Pombe reproduction package.
# Run from anywhere; all paths resolve against the repository root.
"""Train the 6-mer RBF-SVM on the rebuilt benchmark (387 loci).

positives = candidates overlapping the 387 benchmark loci
negatives = candidates overlapping neither benchmark nor fuzzy entries
fuzzy     = excluded from training, evaluated separately
"""

import csv
import random
import re
import sys
import time

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.svm import SVC

sys.stdout.reconfigure(encoding="utf-8")

import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW = str(ROOT_DIR / "data" / "raw")
RES = str(ROOT_DIR / "results")
CANDS = str(ROOT_DIR / "pipeline" / "out")
os.makedirs(RES, exist_ok=True)
LABELS = os.path.join(RES, "candidate_labels.tsv")
LOCI = os.path.join(RES, "benchmark_v2_loci.tsv")
FA = os.path.join(CANDS, "candidates.fasta")

K = 6
SEED = 42

BASES = ["A", "C", "G", "T"]
KMERS = [""]
for _ in range(K):
    KMERS = [m + b for m in KMERS for b in BASES]
IDX = {m: i for i, m in enumerate(KMERS)}


def norm(c):
    c = str(c).replace("chr", "").strip()
    return {"1": "chr1", "2": "chr2", "3": "chr3",
            "I": "chr1", "II": "chr2", "III": "chr3"}.get(c, "chr" + c)


def encode(seq):
    seq = seq.upper()
    v = np.zeros(4 ** K, dtype=np.float32)
    tot = 0
    for i in range(len(seq) - K + 1):
        j = IDX.get(seq[i:i + K])
        if j is not None:
            v[j] += 1.0
            tot += 1
    if tot:
        v /= tot
    return v


cls = {}
for r in csv.DictReader(open(LABELS, encoding="utf-8"), delimiter="\t"):
    cls[r["header"]] = r["class"]
print("candidate classes:", {k: sum(1 for v in cls.values() if v == k)
                             for k in ("positive", "fuzzy", "negative")})

loci = [(norm(r["chrom"]), int(r["start"]), int(r["end"]))
        for r in csv.DictReader(open(LOCI, encoding="utf-8"), delimiter="\t")
        if r["class"] == "benchmark"]
print(f"benchmark loci: {len(loci)}")

pat = re.compile(r"([IV]+|chr\w+):(\d+)-(\d+)")
records, X = [], []
for rec in SeqIO.parse(FA, "fasta"):
    h = rec.description
    seq = str(rec.seq)  # genomic orientation, no reverse complement
    records.append(h)
    X.append(encode(seq))
X = np.vstack(X)
print(f"candidates {len(records)}, dim {X.shape[1]}")

y = np.array([1 if cls[h] == "positive" else 0 for h in records])
fuzzy_idx = [i for i, h in enumerate(records) if cls[h] == "fuzzy"]
pos_idx = [i for i in range(len(records)) if cls[records[i]] == "positive"]
neg_idx = [i for i in range(len(records)) if cls[records[i]] == "negative"]
print(f"train pool: positives {len(pos_idx)}, negatives {len(neg_idx)}, fuzzy {len(fuzzy_idx)}")

samples = [(X[i], 1) for i in pos_idx] + [(X[i], 0) for i in neg_idx]
random.seed(SEED)
random.shuffle(samples)
n_tr = int(len(samples) * 0.8)
tr, te = samples[:n_tr], samples[n_tr:]
Xtr = np.vstack([s[0] for s in tr])
ytr = np.array([s[1] for s in tr])
Xte = np.vstack([s[0] for s in te])
yte = np.array([s[1] for s in te])

t0 = time.time()
clf = SVC(kernel="rbf", class_weight="balanced")
clf.fit(Xtr, ytr)
print(f"trained in {time.time()-t0:.0f}s  (train {len(tr)}, test {len(te)})")

pred_te = clf.predict(Xte)
res = dict(ACC=accuracy_score(yte, pred_te),
           Recall=recall_score(yte, pred_te),
           Precision=precision_score(yte, pred_te),
           F1=f1_score(yte, pred_te),
           AUC=roc_auc_score(yte, clf.decision_function(Xte)))
print("\n=== held-out test set ===")
for k in ("ACC", "Recall", "Precision", "F1", "AUC"):
    print(f"  {k:10s} {res[k]:.4f}")

# genome-wide on all labelled candidates
pred_all = clf.predict(X)
lab_idx = pos_idx + neg_idx
y_all = np.array([y[i] for i in lab_idx])
print("\n=== all labelled candidates (in-sample) ===")
print(f"  ACC {accuracy_score(y_all, pred_all[lab_idx]):.4f}  "
      f"Recall {recall_score(y_all, pred_all[lab_idx]):.4f}  "
      f"Precision {precision_score(y_all, pred_all[lab_idx]):.4f}  "
      f"F1 {f1_score(y_all, pred_all[lab_idx]):.4f}  "
      f"AUC {roc_auc_score(y_all, clf.decision_function(X)[lab_idx]):.4f}")

# high-confidence recall over the 387 benchmark loci
pred_iv = [(norm(pat.search(records[i]).group(1)),
            int(pat.search(records[i]).group(2)),
            int(pat.search(records[i]).group(3)))
           for i in range(len(records)) if pred_all[i] == 1]
cov = 0
for c, s, e in loci:
    L = e - s
    if any(cc == c and (min(x, e) - max(w, s)) > 0
           and ((min(x, e) - max(w, s)) >= 0.5 * L
                or (min(x, e) - max(w, s)) >= 0.7 * (x - w))
           for cc, w, x in pred_iv):
        cov += 1
print(f"\nbenchmark recall: {cov}/{len(loci)} = {cov/len(loci)*100:.2f}%")

# fuzzy candidates: how often called positive
fz = float(np.mean([pred_all[i] for i in fuzzy_idx]))
print(f"fuzzy candidates called positive: {fz*100:.1f}%  (n={len(fuzzy_idx)})")
pos_rate = float(np.mean([pred_all[i] for i in pos_idx]))
neg_rate = float(np.mean([pred_all[i] for i in neg_idx]))
print(f"positive candidates called positive: {pos_rate*100:.1f}%")
print(f"negative candidates called positive: {neg_rate*100:.1f}%")
print(f"\ntotal predicted positives: {int(pred_all.sum())}")
