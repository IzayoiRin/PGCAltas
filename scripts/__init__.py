import os
from build_db import DBuilding

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, 'dataset', 'raw_data')
EMB_DATA = 'EMTAB6967:genes.tsv,clusters.txt,stages.txt,cells.txt,raw_counts.mtx,badcell.txt'


def main():
    flag = os.path.join(os.path.join(BASE_DIR, 'PGCAltas'), 'built_.conf')
    if os.path.exists(flag):
        return
    dbu = DBuilding(DATASET_ROOT, EMB_DATA)
    dbu.load_genes()
    dbu.load_clusters()
    dbu.load_stages()
    dbu.load_cells()
    dbu.load_expression()
    with open(flag, 'w') as f:
        import time
        f.write('DATABASE INSERTED: @%s' % time.ctime())


if __name__ == '__main__':
    main()
