import numpy as np
import pandas as pd

from PGCAltas.utils.StatExpr.analysis_imp import EIMAnalysis
from PGCAltas.utils.StatExpr.cal_imp_area import GenericEIMProcess
from embdata.misc.reader import BinomialDataReader
from embdata.models import GenesInfo, CellsInfo
from ..data_const import POS, NEG, FILE_TYPE


class EMBTABinomalEIMProcess(GenericEIMProcess):

    data_reader_class = BinomialDataReader
    test_size = 0.6

    def __init__(self, filename):
        super().__init__(filename, pos=POS + FILE_TYPE, neg=NEG + FILE_TYPE)

    def get_select_processor_class(self):
        self.screen_processor_class.spilter['test_size'] = self.test_size
        return self.screen_processor_class

    def importance_mtx(self, dimension, split_tt=False):
        self.ppf = self.get_preprocessor()
        self.ppf(self.preprocesses, categories='auto')
        features = self.reader.features
        self.slp = self.get_screen_processor()
        self.slp(self.screen_process,
                 mparams=self.screen_process_params,
                 split=split_tt)
        mtx = self._imp_mtx_processing(features, dimension)
        return mtx


class EMBTABinomalEIMAnalysis(EIMAnalysis):

    data_reader_class = BinomialDataReader
    dimensions = ['binomial', ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def iter_genes(self):
        g_qs = GenesInfo.query.all().order_by('id')
        return iter(g_qs)

    def mark_sig_score(self, sigsco):
        g_qs = self.iter_genes
        gid2name = np.array([[q.id, q.name] for q in g_qs])

        sigsco.IDX = sigsco.IDX.astype(int)
        gid = sigsco.GENE.to_numpy(dtype=int)
        gname = gid2name[gid, 1]
        sigsco.loc[:, 'GENE'] = gname
        return sigsco

    def mark_sig_dataset(self, sigds, sid):
        a = self.reader.samples
        qs = CellsInfo.query.in_bulk(list(a))
        ctype = np.array([qs[int(cid)].type.name for cid in a])
        merge = np.concatenate([self.reader.labels.reshape(-1, 1), ctype.reshape(-1, 1), sigds], axis=1)
        title = np.hstack([['label', 'ctype'], sid])
        # title = np.insert(sid.astype(np.str), 0, 'label')
        return merge, None, title
