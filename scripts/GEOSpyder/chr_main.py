import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.GEOSpyder.ChrFasterDownloader import ChrFasterDownloader

WIDTH = 600
GUIDER = 'spider_guider.txt'
FLAG = 'chrX'
PRIMMER = r'CCAGTAGCAGCACACAGCACTTA'


class DNASequenceTransform(object):

    def __init__(self, seq):
        self.seq = self._is_dna(seq)
        self._seq = self.seq

    def _is_dna(self, seq):
        if isinstance(seq, str):
            pattern = re.compile(r'^[ATGCN]+$', re.IGNORECASE)
            re_result = re.search(pattern, seq, flags=0)
            if re_result:
                return re_result.group().upper()
            else:
                return None
        else:
            raise ValueError()

    def complement(self):
        if self.seq is None:
            return
        self.seq = self.seq.replace('A', 't')\
            .replace('T', 'a')\
            .replace('G', 'c')\
            .replace('C', 'g')\
            .upper()
        return self

    def reverse(self):
        if self.seq is None:
            return
        self.seq = self.seq[::-1].upper()
        return self


def as_fasta():
    dl = ChrFasterDownloader(WIDTH, save_dir='data')
    df = pd.read_csv(GUIDER, sep='\t', header=0, index_col=None)
    df = df[df.chr == FLAG]
    locs = df.location.copy().to_numpy(dtype=np.int32)
    bar = tqdm(enumerate(locs), total=locs.shape[0])
    for idx, i in bar:
        bar.set_description_str('Now Get Seq: {name}'.format(name=i))
        dl(i)
    dl.write()


def transform_fasta():
    dl = ChrFasterDownloader(WIDTH, save_dir='data')
    with open(dl.save_file, 'r') as f:
        lines = f.readlines()
    seq_idxs = list(range(1, len(lines), 2))
    bar = tqdm(enumerate(seq_idxs), total=len(seq_idxs))
    for idx, i in bar:
        seq = lines[i].strip()
        if re.search(PRIMMER, seq):
            bar.set_description_str("Transforming the %sth Sequence" % idx)
            h = DNASequenceTransform(seq)
            lines[i] = h.complement().reverse().seq + "\n"
    with open(dl.save_file, 'w') as f:
        f.writelines(lines)
    print("Transformed")


if __name__ == '__main__':
    # main()
    transform_fasta()
