import numpy as np
import pandas as pd

import pickle

from embdata.misc import *
from embdata.models import CellsInfo, Expression
from PGCAltas.utils.statUniversal import split_arr


# def g_select(gpool_max):
#     patched_func = None
#     g_pool = pool.Pool(gpool_max)
#     g = list()
#     queue = list()
#
#     def inner(instance, *args, **kwargs):
#         setattr(instance, 'g', g)
#         setattr(instance, 'queue', queue)
#         patched_func(instance, *args, **kwargs)
#         for task, arg in g:
#             g_pool.apply_async(task, args=arg)
#         g_pool.join()
#         print(len(queue))
#         return queue
#
#     def g_patch(func):
#         nonlocal patched_func
#         patched_func = func
#         return inner
#
#     return g_patch


class DataGuider(object):

    flags = ('pos', 'neg')
    whole_genes = 29452
    PKL_DIR = 'pickles'
    CSV_DIR = 'texts'

    def __init__(self, file, db_parttion=None):
        # file_path
        self.file = os.path.join(ROOT_, file)
        self.pkl_file = os.path.join(ROOT_, self.PKL_DIR, file).rsplit('.', 1)[0]
        self.csv_file = os.path.join(ROOT_, self.CSV_DIR, file).rsplit('.', 1)[0]
        # built_df
        self.datframe = None
        self._numpy = None
        self._index = None
        self._header = None

        self.db_partition = db_parttion or settings.EXPR_DB_PARTITION
        self.cell_guider = None
        self._expr_dict = dict()

    def build_datframe(self, datframe=None, **kwtable):
        if isinstance(datframe, pd.DataFrame):
            self.datframe = datframe.copy()
        else:
            self.datframe = pd.read_table(self.file, **kwtable)

        temp = [self.datframe.loc[self.datframe.flag == i]
                    .iloc[:, 0:-1] for i in self.flags]

        self._numpy = [i.to_numpy(dtype=np.int32) for i in temp]
        self._index = [i.index.to_numpy(dtype=np.int32) for i in temp]
        self._header = temp[0].columns.to_numpy(dtype=np.int32)
        return self

    def melt_guider(self, flag):

        _numpy = self._numpy[flag]
        indices = self._index[flag]
        columns = self._header

        row, col = _numpy.shape
        cell_guider = np.zeros((row * col, 3), dtype=np.int32)
        idx = 0

        for r in range(row):
            x = indices[r]
            for c in range(col):
                y = columns[c]
                z = _numpy[r, c]
                cell_guider[idx, :] = [x, y, z]
                idx += 1

        return cell_guider[cell_guider[:, 2] > 0]

    @property
    def _guiders(self):
        return {k: self.melt_guider(i) for i, k in enumerate(self.flags)}

    # @g_select(5)
    def select_cell_from_guiders(self):
        """
        :return: {
                    'neg': {
                            "type1": [cid, ...],
                            },
                    'pos': {"type1": [cid, ...],
                            }
                }
        """
        ret = {k: dict() for k in self.flags}

        for k, v in self._guiders.items():
            for x, y, z in v:
                # TODO: Python3.7 choice can't used to queryset generator, plz use first qs
                # qs = np.random.choice(list(CellsInfo.query.filter(type_id=x, stage_id=y)), size=z, replace=False)
                qs = np.random.choice(CellsInfo.query.filter(type_id=x, stage_id=y), size=z, replace=False)

                # db partition
                if ret[k].get(x, None):
                    ret[k][x].extend([q.id for q in qs])
                else:
                    ret[k][x] = [q.id for q in qs]

        self._2pickle(ret, 'select_dict')
        self.cell_guider = ret

    def build_expr_set(self, pkl_name='select_dict', fold=10):
        if self.cell_guider is None:
            self._load_from_pickle(pkl_name, 'cell_guider')

        n = sum([len(d) for d in self.cell_guider.values()])
        print("Total Queried Types: %d" % n)

        for k, v in self.cell_guider.items():
            self._expr_dict[k] = list()
            for ctype, cids in v.items():
                print("CellTypeId: %s\tSelected: %s" % (ctype, len(cids)))
                arrs = split_arr(cids, fold)
                for idx, splited_cids in enumerate(arrs):
                    print("|---Batch: %s / %s" % (idx+1, len(arrs)))
                    mtx_df = self._build_part_expr(ctype, splited_cids)
                    self._expr_dict[k].append(mtx_df)

        self._2pickle(self._expr_dict, 'expr_dict')

    def _build_part_expr(self, ctype, cids):
        mtx = np.zeros([len(cids), self.whole_genes])
        qs = Expression.query.filter(ctype=ctype, cid_id__in=cids)

        row_names = list()
        cur_c, row = -1, -1

        for q in qs:
            c, g, e = q.cid_id, q.gid_id-1, q.expr
            if c != cur_c:
                row_names.append(c)
                row += 1
                cur_c = c
            mtx[row][g] = e

        return pd.DataFrame(mtx, index=row_names)

    @property
    def expr(self):
        if hasattr(self, '__expr'):
            return getattr(self, '__expr')

        if not self._expr_dict:
            self._load_from_pickle('expr_dict', '_expr_dict')

        ret = dict()
        for k, v in self._expr_dict.items():
            if len(v) == 0:
                continue
            if len(v) == 1:
                temp = v[0]
            else:
                temp = pd.concat(v)
            ret[k] = temp.T

        setattr(self, '__expr', ret)
        return ret

    def save(self, ftype=None, **kwargs):
        if ftype == 'txt':
            for flag, datframe in self.expr.items():
                path = '_'.join([self.csv_file, 'Expr%s.txt' % flag.upper()])
                datframe.to_csv(path, **kwargs)
        else:
            self._2pickle(self, self.__class__.__name__)

    def _2pickle(self, obj, filename):
        path = '_'.join([self.pkl_file, filename + '.pkl'])
        if os.path.exists(path):
            pass
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def _load_from_pickle(self, name, attr=None):
        path = '_'.join([self.pkl_file, name + '.pkl'])
        with open(path, 'rb') as f:
            if attr is None:
                return pickle.load(f)
            setattr(self, attr, pickle.load(f))

    def __str__(self):
        return "Guider@%s" % self.file


def guiding(guider='SVMsetGuider2.txt', fold=100, index_col=0, header=0, sep='\t'):
    guider = DataGuider(guider).build_datframe(index_col=index_col, header=header)
    print("Selecting From %s" % guider)
    guider.select_cell_from_guiders()
    print("Building Expression Matrix")
    guider.build_expr_set(fold=fold)
    guider.save('txt', header=header is not None, index=header is not None, sep=sep)
    print("Done!")
