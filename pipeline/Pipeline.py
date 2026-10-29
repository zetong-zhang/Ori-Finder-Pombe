try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    # scikit-learn-intelex is an optional accelerator; the pipeline runs without it
    pass

import time
from subprocess import run, DEVNULL
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from scipy.ndimage import gaussian_filter1d
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks
from Bio import SeqIO
from collections import defaultdict
import joblib

SOFTWARE = 'Ori-Finder-Pombe'

""" Util Functions """
def kmer_vector(
    seq: str, k: int, 
    normalized: bool = True, 
    alphabet: str = "ACGT"
) -> np.ndarray:
    seq = seq.upper()
    base = len(alphabet)
    total_kmers = base ** k
    if k > len(seq):
        return np.zeros(total_kmers, dtype=np.float64)
    trans = {c: i for i, c in enumerate(alphabet)}
    arr = np.array([trans.get(ch, -1) for ch in seq], dtype=np.int8)
    windows = sliding_window_view(arr, k)
    valid = (windows != -1).all(axis=1)
    if not np.any(valid):
        return np.zeros(total_kmers, dtype=np.float64)
    valid_windows = windows[valid].astype(np.uint8)
    weights = base ** np.arange(k - 1, -1, -1)
    codes = valid_windows @ weights
    vec = np.bincount(codes, minlength=total_kmers).astype(np.float64)
    if normalized:
        total = vec.sum()
        if total > 0: vec /= total
    return vec

def at_curve(
    seq: str, window: int = 101, 
    sigma: float = 2, at_thres: float = 0.75
):
    seq_bytes = seq.encode()
    L = len(seq)

    table = bytearray(256)
    table[ord('A')] = 1
    table[ord('T')] = 1
    table = bytes(table)

    pad = (window + 1) // 2
    padded = b'N' * pad + seq_bytes + b'N' * pad
    at_bytes = padded.translate(table)
    at_arr = np.frombuffer(at_bytes, dtype=np.uint8)

    prefix = np.concatenate(([0], np.cumsum(at_arr)))
    
    half_left = window // 2
    half_right = (window + 1) // 2
    i_arr = np.arange(L)
    s_idx = i_arr + pad - half_left
    e_idx = i_arr + pad + half_right
    counts = prefix[e_idx] - prefix[s_idx]
    raw = counts / window

    n_high = len(list(filter(lambda x:x >= at_thres, raw)))
    filtered = gaussian_filter1d(raw, sigma)

    return filtered, n_high

def find_at_rich(
    seq: str, window: int = 1000, 
    peak_h: float = 0.55, 
):
    L = len(seq)
    half_left = window // 2
    half_right = (window + 1) // 2
    filtered, _ = at_curve(seq=seq, window=window)
    peaks, _ = find_peaks(filtered, height=peak_h, distance=window)
    
    peak_regions = []
    for i in peaks:
        start = i - half_left
        end = i + half_right
        if start >= 0 and end <= L:
            peak_regions.append((start, end, None))
    
    return filtered, peak_regions

ROOT = Path(__file__).parent
DATABASE = ROOT / 'database'
REF_FNA = str(DATABASE / 'fna' / 'GCF_000002945.1.fna')
REF_GFF = str(DATABASE / 'gff' / 'GCF_000002945.1.gff')
MODEL_PATH = DATABASE / 'mod' / 'pombe_svm.joblib'

MIN_CAND_LEN = 500

CEN_MOTIFS = [
    "ATATAGAG", 
    "CAGTACTAGTGT", 
    "TACTGACACTAG", 
    "TAGTGG", 
    "TAGTTATAGTTG", 
    "TCTCTATA", 
    "TTATAGTTGTAG", 
    "ATTGATATGGAC",
    "CATTATATA",
    "AGTGGT"
]

# Max gap (bp) between consecutive tandem repeats to be clustered
# into the same centromere; isolated outliers are discarded.
MAX_CEN_TR_GAP = 70000

@dataclass
class Region:
    host: str
    start: int
    end: int
    type: str    = 'region'
    strand: str  = '.'
    attrs: str   = '.'
    seq: str     = ""
    score: float = 0
    name: str    = ''

    def __post_init__(self):
        self.qlfs = dict()
    
    def find_a_string(self):
        if not self.seq or len(self.seq) < 15:
            return
        seq = self.seq.upper()
        a = seq[:15].count('A')
        t = seq[:15].count('T')
        total = a + t
        best_skew = (a - t) / total if total else 0
        for i in range(1, len(seq) - 14):
            out = seq[i - 1]
            inc = seq[i + 14]
            if out == 'A': a -= 1
            elif out == 'T': t -= 1
            if inc == 'A': a += 1
            elif inc == 'T': t += 1
            total = a + t
            if total == 0: continue
            skew = (a - t) / total
            if abs(skew) > abs(best_skew):
                best_skew = skew
        self.strand = '+' if best_skew >= 0 else '-'
    
    @property
    def length(self):
        return self.end - self.start
    def gff(self) -> str:
        attrs = self.attrs if self.attrs and self.attrs != '.' else ''
        if self.name and 'Name=' not in attrs:
            attrs += f';Name={self.name}' if attrs else f'Name={self.name}'
        attributes = attrs if attrs else '.'
        return f'{self.host}\t{SOFTWARE}\t{self.type}\t\
{self.start}\t{self.end}\t{self.score:.3f}\t{self.strand}\t.\t{attributes}'


@dataclass
class Pipeline:
    target: Optional[Path] = None
    status: str = ''
    mode: str = 'Genome'
    genes: dict = None
    error: str = None
    window: int = 1000
    min_at: float = 0.55
    log: list = None
    origins: dict = None
    thresh: float = 0.0

    def __post_init__(self):
        self.log = []

    def add_log(self, msg: str):
        ts = time.strftime('%H:%M:%S')
        self.log.append(f'[{ts}] {msg}')

    @staticmethod
    def parse_gff(gff_path: str) -> List[Region]:
        feat_types = {'gene', 'rep_origin'}
        genes = defaultdict(list)
        with open(gff_path, 'r') as handle:
            for line in handle:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 9: continue
                seqid, _, feat_type, start, end, _, strand, _, attrs = parts
                if feat_type in feat_types:
                    name = ''
                    for kv in attrs.split(';'):
                        if kv.startswith('Name='):
                            name = kv[5:]
                            break
                    genes[seqid].append(Region(
                        host=seqid,
                        start=int(start),
                        end=int(end),
                        type=feat_type,
                        strand=strand,
                        attrs=attrs,
                        name=name,
                    ))
        return genes

    def liftoff_annotate(self):
        self.add_log('Annotating genes by Liftoff ...')

        import shutil
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix='orifinder_liftoff_')
        output_gff = str(Path(tmpdir) / f'{self.target.stem}.liftoff.gff')
        cmd = ['liftoff', '-g', REF_GFF, '-o', output_gff, str(self.target), REF_FNA]
        try:
            run(cmd, check=True, stderr=DEVNULL)
            self.genes = self.parse_gff(output_gff)
            total = sum(len(v) for v in self.genes.values())
            self.add_log(f'Parsed {total} genes from Liftoff output')
        except Exception as e:
            self.add_log(f'Liftoff failed: {e}')
            self.add_log('Continuing without gene annotation')
            self.genes = None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def trf_scan(self):
        self.add_log(f'Predicting centromeres by TRF ...')

        fna_txt = ""
        for idx, record in enumerate(self.records):
            fna_txt += f">{idx:06d}\n{record['seq']}\n"
        trf_cmd = [
            "trf", "-",
            "2", "7", "7", "80", "10", "50", "200",
            "-h", "-d", "-ngs"
        ]
        result = run(
            trf_cmd,
            input=fna_txt,
            text=True,
            capture_output=True
        )
        cur_idx = None
        if result.returncode == 0:
            trf_outputs = result.stdout.splitlines()
            for line in trf_outputs:
                if line.startswith("@"):
                    cur_idx = int(line[1:].split()[0])
                    self.records[cur_idx]['trs'] = []
                    continue
                items = line.split()
                consensus = items[13]
                if consensus in CEN_MOTIFS:
                    self.records[cur_idx]['trs'].append({
                        "start": int(items[0]),
                        "end": int(items[1]),
                        "motif": consensus
                    })
        for idx, record in enumerate(self.records):
            trs = record.get('trs', [])
            if len(trs) > 0:
                trs = sorted(trs, key=lambda t: t['start'])
                record['trs'] = trs  # keep all repeats for display
                # Cluster repeats by proximity. Every cluster of >=2 repeats is
                # treated as a centromeric region; isolated singleton repeats are
                # dropped, unless the contig has no multi-repeat cluster at all.
                clusters = [[trs[0]]]
                for tr in trs[1:]:
                    if tr['start'] - clusters[-1][-1]['end'] <= MAX_CEN_TR_GAP:
                        clusters[-1].append(tr)
                    else:
                        clusters.append([tr])
                regions = [c for c in clusters if len(c) >= 2] or [clusters[0]]
                record['cen'] = [[c[0]['start'], c[-1]['end']] for c in regions]
                for cen_start, cen_end in record['cen']:
                    self.add_log(f"- Centromere #{idx+1}  [{record['id']}:{cen_start}..{cen_end}]")

    def find_candidates(self, tol: int=30):
        self.add_log(f'Detecting AT-rich regions ...')

        self.candidates = defaultdict(list)
        n_cand = 0
        for record in self.records:
            vseq = list('N'*record['len'])
            curve, islands = find_at_rich(record['seq'], window=self.window, peak_h=self.min_at)
            
            for start, end, _ in islands:
                vseq[start:end] = 'Z' * (end - start)
                
            if self.genes:
                for gene in self.genes.get(record['id'], []):
                    start, end = gene.start, gene.end
                    vseq[start:end] = 'N' * (end - start)
            if record.get('cen', None):
                for cen_start, cen_end in record['cen']:
                    vseq[cen_start:cen_end] = 'N' * (cen_end - cen_start)
            
            recording, last, mismatch = False, -1, 0
            for idx, s in enumerate(vseq):
                if s != 'N' and not recording:
                    last = idx; recording = True; mismatch = 0
                elif recording:
                    if s == 'N': mismatch += 1
                    else: mismatch = 0
                    
                    if mismatch >= tol:
                        sublen = idx - last
                        score = curve[last] if last <= len(curve) else curve[-1]
                        if sublen >= MIN_CAND_LEN and score > 0.550:
                            end = idx - mismatch + 1
                            subseq = record['seq'][last:end]
                            region = Region(host=record['id'], start=last, end=end,
                                            type='AT_rich', score=-1.0, seq=subseq)
                            region.find_a_string()

                            sub_curve, n_high = at_curve(subseq)

                            region.qlfs['cx'] = list(range(region.length))
                            region.qlfs['cy'] = sub_curve.tolist()
                            region.qlfs['nh'] = n_high

                            self.candidates[record['id']].append(region)
                            n_cand += 1
                        recording = False

        n_record = len(self.records)
        self.add_log(f'Found {n_cand} AT-rich regions on {n_record} sequences ...')
    
    def svm_predict(self):
        self.add_log('Running SVM prediction...')

        self.origins = defaultdict(list)
        if not self.candidates: return
        
        clf = joblib.load(str(MODEL_PATH))
        self.origins = defaultdict(list)
        for chrom, cands in self.candidates.items():
            seqs = []
            for r in cands:
                # Candidates are encoded in their genomic orientation. The model is
                # trained on forward-strand k-mer frequencies, so minus-strand
                # intervals are NOT reverse-complemented here (see METHODS).
                s = r.seq.upper()
                seqs.append(s)
            if not seqs: continue
            X = np.vstack([kmer_vector(s, 6) for s in seqs])
            scores = clf.decision_function(X)
            for r, score in zip(cands, scores):
                r.score = score
                if score >= self.thresh:
                    r.type = 'ARS'
                    self.origins[chrom].append(r)
        total = sum(len(v) for v in self.origins.values())
        self.add_log(f'SVM predicted {total} origins')
    
    def run(self):
        try:
            self.add_log('Pipeline started')
            self.records = [{'id': record.id, 'seq': str(record.seq).upper(), 'len': len(record.seq)}
                            for record in SeqIO.parse(str(self.target), "fasta")]
            if self.mode == 'Single':
                for rec in self.records:
                    if rec['len'] > 6000:
                        raise ValueError(f'Sequence "{rec["id"]}" exceeds 6000 bp ({len(rec["seq"])} bp)')
                self.candidates = defaultdict(list)
                for rec in self.records:
                    curve, _ = at_curve(rec['seq'])
                    region = Region(
                        host=rec['id'], start=1, end=rec['len'],
                        type='cand', seq=rec['seq'],
                    )
                    region.find_a_string()
                    region.qlfs['cx'] = list(range(1, rec['len'] + 1))
                    region.qlfs['cy'] = curve.tolist()
                    self.candidates[rec['id']].append(region)
                self.svm_predict()
            else:
                if not self.genes: self.liftoff_annotate()
                else: self.add_log('Using custom GFF annotation')
                self.trf_scan()
                self.find_candidates()
                self.svm_predict()
            self.status = 'Completed'
            self.add_log('Pipeline completed successfully')
        except Exception as e:
            self.status = 'Error'
            self.error = str(e)
