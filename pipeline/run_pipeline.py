# -*- coding: utf-8 -*-
"""Run the Ori-Finder-Pombe origin-annotation pipeline on a fission-yeast assembly.

Every assembly — including the bundled reference genome — takes the same full
``Genome`` pipeline: Liftoff gene annotation, TRF centromere prediction, AT-rich
candidate extraction and 6-mer SVM origin prediction.

Usage
-----
    python pipeline/run_pipeline.py -i assembly.fna -o pipeline/out   # arbitrary assembly
    python pipeline/run_pipeline.py -o pipeline/out                   # bundled reference genome
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = HERE / "database"
REF_FNA = DB / "fna" / "GCF_000002945.1.fna"
sys.path.insert(0, str(HERE))

import Pipeline as P  # noqa: E402


def write_fasta(regions, path):
    with open(path, "w") as f:
        for chrom in sorted(regions or {}):
            for r in regions[chrom]:
                f.write(f">{r.type} {chrom}:{r.start}-{r.end} ({r.strand})\n")
                for i in range(0, len(r.seq), 80):
                    f.write(r.seq[i:i + 80] + "\n")


def write_gff(regions, path):
    with open(path, "w") as f:
        f.write("##gff-version 3\n")
        for chrom in sorted(regions or {}):
            for r in regions[chrom]:
                f.write(r.gff() + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Annotate DNA replication origins in a fission-yeast assembly.")
    ap.add_argument("-i", "--input", default=str(REF_FNA),
                    help="input FASTA assembly; defaults to the bundled reference genome")
    ap.add_argument("-o", "--out", default=str(HERE / "out"))
    ap.add_argument("--thresh", type=float, default=0.0,
                    help="SVM decision threshold for calling an origin (default 0.0)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pipe = P.Pipeline(target=Path(args.input), mode="Genome", thresh=args.thresh)
    pipe.run()
    if pipe.status != "Completed":
        print(f"pipeline failed: {pipe.error}", file=sys.stderr)
        sys.exit(1)

    write_fasta(pipe.candidates, out / "candidates.fasta")
    write_fasta(pipe.origins, out / "origins.fasta")
    write_gff(pipe.candidates, out / "candidates.gff")
    write_gff(pipe.origins, out / "origins.gff")
    if pipe.genes:
        write_gff(pipe.genes, out / "genes.gff")

    n_cand = sum(len(v) for v in pipe.candidates.values())
    n_ori = sum(len(v) for v in pipe.origins.values())
    print(f"candidates: {n_cand}")
    print(f"predicted origins: {n_ori}")
    print(f"written: {out / 'candidates.gff'}")
    print(f"written: {out / 'candidates.fasta'}")
    print(f"written: {out / 'origins.gff'}")
    print(f"written: {out / 'origins.fasta'}")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
