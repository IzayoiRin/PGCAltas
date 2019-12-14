import os
import django

if os.environ.get("DJANGO_SETTINGS_MODULE", None) is None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "PGCAltas.settings.dev")
    django.setup()

from django.conf import LazySettings
from embdata import models as emb
from scripts import const as c

settings = LazySettings()
PATH_ = os.path.dirname(settings.BASE_DIR) + settings.DATASET_URL


class DBuilding(object):

    def __init__(self, root, data: str):
        data_dir, data = data.split(':')
        genes, clusters, stages, cells, expression, bad = data.split(',')
        self.root = os.path.join(root, data_dir, c.RAW_DATA_DIR)
        self.genes = os.path.join(self.root, genes)
        self.clusters = os.path.join(self.root, clusters)
        self.stages = os.path.join(self.root, stages)
        self.cells = os.path.join(self.root, cells)
        self.expression = os.path.join(self.root, expression)
        self.badcell = os.path.join(self.root, bad)

    def load_genes(self):
        with open(self.genes, 'r') as f:
            while True:
                line = f.readline().rstrip()
                if not line:
                    break
                d = dict(zip(('access', 'name'), line.split('\t')))
                emb.GenesInfo(**d).save()

    def load_clusters(self):
        func = lambda line: line.rstrip().split('\t')[0]
        bulk_list = self._load_single_column(emb.ClustersInfo, func, name=self.clusters)
        emb.ClustersInfo.query.bulk_create(bulk_list)

    def load_stages(self):
        func = lambda line: line.rstrip().split('\t')[0]
        bulk_list = self._load_single_column(emb.StagesInfo, func, stage=self.stages)
        emb.StagesInfo.query.bulk_create(bulk_list)

    @staticmethod
    def _load_single_column(table, func, **fieldkwarg):
        """
        create single column bulk_list
        :param table: tb_orm
        :param feildkwarg: field : data_path
        :return:
        """
        field = list(fieldkwarg.keys())[0]
        path = fieldkwarg[field]
        with open(path, 'r') as f:
            lines = f.readlines()
        return [table(
            **{field: func(line)}
        ) for line in lines]

    def load_cells(self):
        fields = ['name', 'barcode', 'sample', 'stage_id', 'type_id']
        with open(self.cells, 'r') as f:
            line = f.readline()
            while line:
                bulk_list = list()
                for _ in range(10000):
                    line = f.readline()
                    if not line:
                        break
                    ps = dict(zip(fields, line.rstrip().split('\t')))
                    bulk_list.append(emb.CellsInfo(**ps))
                emb.CellsInfo.query.bulk_create(bulk_list)

    def load_expression(self):
        badcell = self._get_badcell()
        fields = ['gid_id', 'cid_id', 'expr', 'ctype']
        last, tid, cid = None, None, None
        with open(self.expression, 'r') as f:
            for _ in range(2):
                line = f.readline()
            while line:
                bulk_list = list()
                for _ in range(100000):
                    line = f.readline()
                    if not line:
                        break
                    context = line.rstrip().split(' ')
                    if context[1] in badcell:
                        continue
                    if context[1] != last:
                        last = context[1]
                        cell = emb.CellsInfo.query.get(name=context[1])
                        tid, cid = cell.type_id, cell.id
                    context[1] = cid
                    context.append(tid)
                    ps = dict(zip(fields, context))
                    bulk_list.append(emb.Expression(**ps))
                emb.Expression.query.bulk_create(bulk_list)

    def _get_badcell(self):
        badcell = list()
        with open(self.badcell, 'r') as f:
            line = f.readline()
            while line:
                line = f.readline()
                a = line.rstrip().split('\t')[0]
                badcell.append(a)
        return badcell


def db():
    flag = os.path.join(settings.BASE_DIR, 'built_.conf')
    if os.path.exists(flag):
        with open(flag, 'r') as f:
            print(f.read())
        return
    dbu = DBuilding(PATH_, c.EMB_DATA)
    dbu.load_genes()
    dbu.load_clusters()
    dbu.load_stages()
    dbu.load_cells()
    dbu.load_expression()
    with open(flag, 'w') as f:
        import time
        f.write('DATABASE INSERTED: @%s\n'
                'Do NOT DELETE this config UNLESS need to FLUSH database' % time.ctime())
