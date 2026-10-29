# -*- coding: utf-8 -*-
# Part of the Ori-Finder-Pombe reproduction package.
# Run from anywhere; all paths resolve against the repository root.
"""Reproduce exactly 387 benchmark loci and 202 fuzzy candidates.

Rule (the one used for the 387 figure):
  - loci = oriDB (oriDB.bed, chr*NO removed) + OM merged by direct overlap
    -> 518 loci (254 both, 215 oriDB-only, 49 OM-only)
  - a locus is Pu-supported when ANY Pu-seq site (>=20%) has its 300-bp bin
    centre within 900 bp of the locus (no exclusive assignment)
  - benchmark = loci supported by >= 2 sources
  - Pu-seq sites not within 900 bp of any locus form additional (fuzzy) loci
"""

import csv
import gzip
import re
import sys
from collections import defaultdict

from Bio import SeqIO

sys.stdout.reconfigure(encoding="utf-8")

import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW = str(ROOT_DIR / "data" / "raw")
RES = str(ROOT_DIR / "results")
CANDS = str(ROOT_DIR / "pipeline" / "out")
os.makedirs(RES, exist_ok=True)
TOL = 900


def norm(c):
    c = str(c).replace("chr", "").strip()
    return {"1": "chr1", "2": "chr2", "3": "chr3",
            "I": "chr1", "II": "chr2", "III": "chr3"}.get(c, "chr" + c)


raw = [l.split() for l in open(os.path.join(RAW, "oriDB.bed")) if l.strip()]
pos = [(norm(x[0]), int(x[1]), int(x[2])) for x in raw if "NO" not in x[0]]
om = [(norm(r["chromosome"]), int(r["start"]) - 1, int(r["end"]))
      for r in csv.DictReader(open(os.path.join(RAW, "OMsite_position.txt"), encoding="utf-8"),
                              delimiter="\t")]
by = defaultdict(list)
for line in gzip.open(os.path.join(RAW, "GSE62108_origin-effici.bedgraph.gz"), "rt"):
    f = line.split()
    if len(f) < 4:
        continue
    by[norm(f[0])].append((int(f[1]), int(f[2]), float(f[3])))
puseq = [(c, a, b) for c in by for (a, b, v) in by[c] if v >= 20]

# ---- build 518 loci ----
bychr = defaultdict(list)
for c, s, e in pos:
    bychr[c].append((s, e, "od"))
for c, s, e in om:
    bychr[c].append((s, e, "om"))

loci = []
for c, ivs in bychr.items():
    ivs.sort()
    cur = None
    for s, e, src in ivs:
        if cur is None:
            cur = dict(c=c, s=s, e=e, od=(src == "od"), om=(src == "om"), pu=False)
        elif s <= cur["e"]:
            cur["e"] = max(cur["e"], e)
            cur["od"] = cur["od"] or (src == "od")
            cur["om"] = cur["om"] or (src == "om")
        else:
            loci.append(cur)
            cur = dict(c=c, s=s, e=e, od=(src == "od"), om=(src == "om"), pu=False)
    if cur is not None:
        loci.append(cur)

print(f"loci from oriDB+OM: {len(loci)} "
      f"(both {sum(1 for x in loci if x['od'] and x['om'])}, "
      f"oriDB-only {sum(1 for x in loci if x['od'] and not x['om'])}, "
      f"OM-only {sum(1 for x in loci if x['om'] and not x['od'])})")

# ---- Pu support: any site centre within 900 bp ----
pu_by_c = defaultdict(list)
for c, a, b in puseq:
    pu_by_c[c].append((a + b) // 2)

for x in loci:
    x["pu"] = any(x["s"] - TOL <= ctr <= x["e"] + TOL for ctr in pu_by_c[x["c"]])

n_both = sum(1 for x in loci if x["od"] and x["om"])
n_od_pu = sum(1 for x in loci if x["od"] and not x["om"] and x["pu"])
n_om_pu = sum(1 for x in loci if x["om"] and not x["od"] and x["pu"])
n_od = sum(1 for x in loci if x["od"] and not x["om"] and not x["pu"])
n_om = sum(1 for x in loci if x["om"] and not x["od"] and not x["pu"])
print(f"oriDB-only with Pu {n_od_pu}, without {n_od}")
print(f"OM-only with Pu {n_om_pu}, without {n_om}")

bench = [x for x in loci if (x["od"] + x["om"] + x["pu"]) >= 2]
fuzzy = [x for x in loci if (x["od"] + x["om"] + x["pu"]) == 1]
print(f"\nbenchmark loci: {len(bench)}  (both {n_both} + oriDB+Pu {n_od_pu} + OM+Pu {n_om_pu})")
print(f"fuzzy annotation loci: {len(fuzzy)}")

# orphan Pu sites -> extra fuzzy loci
orphan = []
for c, a, b in puseq:
    ctr = (a + b) // 2
    if not any(x["s"] - TOL <= ctr <= x["e"] + TOL for x in loci if x["c"] == c):
        orphan.append((c, a, b))
bychr2 = defaultdict(list)
for c, a, b in orphan:
    bychr2[c].append((a, b))
orphan_loci = 0
for c, ivs in bychr2.items():
    ivs.sort()
    cur_e = None
    for a, b in ivs:
        if cur_e is None or a - cur_e > TOL:
            orphan_loci += 1
            cur_e = b
        else:
            cur_e = max(cur_e, b)
print(f"Pu-seq sites not assigned to any locus: {len(orphan)} -> {orphan_loci} loci")

# ---- candidate labels ----
pat = re.compile(r"([IV]+|chr\w+):(\d+)-(\d+)")
cands = []
for rec in SeqIO.parse(os.path.join(CANDS, "candidates.fasta"), "fasta"):
    h = rec.description
    m = pat.search(h)
    cands.append((h, norm(m.group(1)), int(m.group(2)), int(m.group(3))))


def hits(c, s, e, ivs):
    for cc, a, b in ivs:
        if cc != c:
            continue
        ov = min(e, b) - max(s, a)
        if ov > 0 and (ov >= 0.5 * (b - a) or ov >= 0.7 * (e - s)):
            return True
    return False


biv = [(x["c"], x["s"], x["e"]) for x in bench]
fiv = [(x["c"], x["s"], x["e"]) for x in fuzzy] + \
      [(c, a, b) for c, a, b in orphan]

pos_c = [h for h, c, s, e in cands if hits(c, s, e, biv)]
fz_c = [h for h, c, s, e in cands if h not in set(pos_c) and hits(c, s, e, fiv)]
neg_c = [h for h, c, s, e in cands if h not in set(pos_c) and h not in set(fz_c)]
print(f"\npositive candidates {len(pos_c)}, fuzzy {len(fz_c)}, negative {len(neg_c)}, "
      f"total {len(pos_c)+len(fz_c)+len(neg_c)}")

with open(os.path.join(RES, "candidate_labels.tsv"), "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
    w.writerow(["header", "label", "class"])
    for h in pos_c:
        w.writerow([h, 1, "positive"])
    for h in fz_c:
        w.writerow([h, "", "fuzzy"])
    for h in neg_c:
        w.writerow([h, 0, "negative"])
print("written: candidate_labels.tsv")


# ---- benchmark loci table ----
with open(os.path.join(RES, "benchmark_v2_loci.tsv"), "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
    w.writerow(["chrom", "start", "end", "length", "original_source",
                "puseq_support", "n_sources", "class"])
    for x in sorted(loci, key=lambda y: (y["c"], y["s"])):
        n = int(x["od"]) + int(x["om"]) + int(x["pu"])
        src = "both" if (x["od"] and x["om"]) else ("oriDB" if x["od"] else "site")
        w.writerow([x["c"], x["s"], x["e"], x["e"] - x["s"], src,
                    "yes" if x["pu"] else "no", n,
                    "benchmark" if n >= 2 else "fuzzy"])
print("written: benchmark_v2_loci.tsv")
