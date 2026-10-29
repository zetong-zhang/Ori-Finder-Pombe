# Ori-Finder-Pombe

Genome-wide prediction of DNA replication origins in *Schizosaccharomyces pombe*.
This repository contains the prediction pipeline, the evidence sets used to build the
benchmark, and the scripts that derive the numbers reported in the manuscript.

```
data/raw/     the evidence sources (oriDB, Orc4-Mcm2, Pu-seq)
pipeline/     candidate extraction + SVM prediction, runnable on its own
scripts/      benchmark construction, candidate labelling, training, ablation
```

---

## 1. Data

| File | Content | Source |
|---|---|---|
| `data/raw/oriDB.bed` | oriDB 3.0 entries for *S. pombe*, manually cleaned. Three columns `chromosome, start, end`; entries kept but not used as evidence carry a `NO` suffix on the chromosome name (e.g. `INO`) and are filtered out by the scripts | oriDB |
| `data/raw/OMsite_position.txt` | Orc4-Mcm2 co-localised sites. Columns `chromosome, start, end, summit` | Masuda et al. (2020) |
| `data/raw/GSE62108_origin-effici.bedgraph.gz` | Pu-seq initiation efficiency in 300-bp bins (`chrom, start, end, efficiency%`); 1,145 windows, 0.07-83.8%, of which those >= 20% are used as evidence | Daigaku et al. (2015), GEO GSE62108 |

The pipeline needs the following inputs, kept next to the code in `pipeline/database/`:

| File | Content |
|---|---|
| `fna/GCF_000002945.1.fna` | reference genome, ASM294v2 / GCF_000002945.1, three nuclear chromosomes (12,571,820 bp) |
| `gff/GCF_000002945.1.gff` | gene annotation, 5,061 non-dubious genes (PomBase) |
| `mod/pombe_svm.joblib` | the deployed 6-mer RBF-SVM model |

---

## 2. Predicting origins

```bash
python pipeline/run_pipeline.py -i assembly.fna -o pipeline/out   # arbitrary assembly
python pipeline/run_pipeline.py -o pipeline/out                   # reference genome
```

This runs the code in `pipeline/Pipeline.py`: AT-rich window scanning (1,000-bp window,
1-bp step, Gaussian sigma = 2, `find_peaks(height >= 0.55, distance = 1000)`, plus/minus
500 bp per peak, coding and centromeric regions masked, intervals closer than 30 bp
merged, intervals shorter than 500 bp dropped), 6-mer encoding and SVM scoring; an
interval is called an origin when the decision value is >= 0.

Every assembly — including the bundled reference genome — takes the same full `Genome`
pipeline: gene annotation by Liftoff and centromere prediction by TRF precede the AT-rich
and SVM steps above. This requires the external `liftoff`, `trf` and `minimap2` binaries
(see section 5).

Output in `pipeline/out/`: `candidates.gff`, `candidates.fasta`, `origins.gff`,
`origins.fasta`, and `genes.gff` (gene annotation, when available).

**Expected result on the reference genome: 4,171 candidate regions and 466 predicted
origins** on the three nuclear chromosomes. This repository's `GCF_000002945.1.fna`
contains only the three nuclear chromosomes; the reference strain in the 89-assembly
database additionally includes a mitochondrial contig and yields 467 origins.

---

## 3. Benchmark and training

```bash
python scripts/01_build_benchmark.py      # benchmark loci + candidate labels
python scripts/02_train_svm.py            # 8:2 split, train and evaluate
python scripts/03_architecture_compare.py # six encodings x three classifiers
```

**Benchmark rule.** Three lines of evidence are used: oriDB, Orc4-Mcm2 (OM) and Pu-seq.
oriDB and OM intervals are merged by direct overlap, giving 518 loci (254 supported by
both, 215 by oriDB only, 49 by OM only). A Pu-seq window supports a locus when the centre
of its 300-bp bin lies within 900 bp of the locus. A locus supported by at least two of
the three lines enters the benchmark; a locus supported by a single line is set aside as
fuzzy and is used neither for training nor for the headline metrics. Pu-seq windows
that support no locus become additional fuzzy loci.

**Candidate labels.** A candidate is *positive* when it overlaps a benchmark locus by at
least 50% of the locus or at least 70% of the candidate, *fuzzy* when it overlaps only
a single-source locus, and *negative* otherwise.

**Model.** 6-mer frequencies (4,096 dimensions, normalised by k-mer count) and an
RBF-kernel SVM with balanced class weights. The labelled candidates are split 8:2 with a
fixed seed; the held-out part gives Table 1 of the manuscript, and the genome-wide figures
refer to the 80% model. The model used for prediction was retrained on all labelled
candidates after evaluation.

**Encoding convention.** In the training script and the prediction pipeline, every
candidate is encoded in its genomic orientation; minus-strand intervals are not
reverse-complemented.

### Expected output

`scripts/01_build_benchmark.py`:

```
loci from oriDB+OM: 518 (both 254, oriDB-only 215, OM-only 49)
oriDB-only with Pu 109, without 106
OM-only with Pu 24, without 25
benchmark loci: 387  (both 254 + oriDB+Pu 109 + OM+Pu 24)
fuzzy annotation loci: 131
Pu-seq sites not assigned to any locus: 138 -> 135 loci
positive candidates 382, fuzzy 202, negative 3587, total 4171
```

`results/benchmark_v2_loci.tsv` is written byte for byte as in the manuscript analysis
(MD5 `563913b67005dbace0cfd1d192ba94ea`).

| Quantity | Value |
|---|---|
| non-redundant loci after merging the three sources | 653 |
| benchmark loci (at least two sources) | 387 |
| fuzzy loci (single source: 131 annotation loci + 135 orphan Pu-seq) | 266 |
| candidate sequences: positive / fuzzy / negative | 382 / 202 / 3,587 |
| held-out accuracy / recall / precision (Table 1) | 0.9597 / 0.7692 / 0.8108 |
| genome-wide accuracy / precision (Table 2) | 0.9872 / 0.9169 |
| benchmark recall (359/387, Table 2) | 0.9276 |

`scripts/03_architecture_compare.py` writes `results/arch_compare/`, which holds the
five-fold comparison of six sequence encodings and three classifiers quoted in
Supplementary Table S2.

---

## 4. Cross-strain database

Applying the pipeline to 89 *S. pombe* assemblies (48 strains) yields **42,172 predicted
origins** in total, and the 387 benchmark loci fall into the conservation tiers **201
core / 70 highly conserved / 101 medium conserved / 7 weakly conserved / 8 uncovered**.

---

## 5. Environment

Python 3.10+:

```
numpy
scipy
scikit-learn
biopython
joblib
```

`pip install -r requirements.txt`

`lightgbm` is used only by `03_architecture_compare.py` as its third classifier. It is
optional: without it that script falls back to RBF-SVM and RF. The conda command below
installs it; with pip, add it separately (`pip install lightgbm`).

Every `run_pipeline.py` run — including the bundled reference genome — requires the
external `liftoff` (with `minimap2`) and `trf` binaries on `PATH`.

### Installing with conda

The simplest way to get the Python stack and the three external binaries into one
environment is conda (conda-forge + bioconda):

```bash
conda create -n orifinder -c conda-forge -c bioconda \
    python=3.10 numpy scipy scikit-learn biopython joblib lightgbm \
    liftoff minimap2 trf
conda activate orifinder
```

If the binaries already live in an existing conda environment, activate that one
instead and skip the `conda create` step:

```bash
conda activate <your-env>
```

After activation, confirm that all three binaries are visible:

```bash
which liftoff minimap2 trf
```

The pipeline is then run as described in section 2.

The scripts resolve every path against the repository root and can be run from any
directory.

---
