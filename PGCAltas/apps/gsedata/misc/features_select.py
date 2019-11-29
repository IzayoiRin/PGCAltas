import numpy as np
import pandas as pd

from PGCAltas.utils.StatExpr.analysis_imp import EIMAnalysis
from PGCAltas.utils.StatExpr.cal_imp_area import EIMProcess
from gsedata.misc.reader import ReaderFromDimensions


class GSEBinDimensionEIMProcess(EIMProcess):

    data_reader_class = ReaderFromDimensions

    def __init__(self, filename):
        super().__init__(filename)

    def get_preprocessor(self):
        ppf_cls = self.preprocessor_class()
        ppf = ppf_cls.init_from_data(self.reader.dataset, (self.reader.tlabels, self.reader.llabels))
        return ppf

    def get_select_processor(self):
        slp_cls = self.get_select_processor_class()
        dataset, tlabels, llabels = self.ppf.dataset, self.ppf.labels[0], self.ppf.labels[1],
        slp = slp_cls().init_from_data(dataset, (tlabels, llabels))
        return slp

    def importance_mtx(self, dimension, split_tt=False):
        self.ppf = self.get_preprocessor()
        self.ppf(*self.preprocesses, dim=dimension)
        features = self.reader.features
        self.slp = self.get_select_processor()
        # RDF_PARAMS in const.py
        self.slp(self.select_process,
                 mparams=self.select_process_params,
                 dim=dimension, split_tt=split_tt)

        mtx = self._imp_mtx_processing(features, dimension)
        return mtx


class GSEBinDimensionEIMAnalysis(EIMAnalysis):

    data_reader_class = ReaderFromDimensions
    dimensions = ['time', 'loc']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mark_sig_dataset(self, sigds, sid):
        tl, ll = self.reader.tlabels, self.reader.llabels
        layer = np.array([s[0] for s in self.reader.samples])
        merge = np.concatenate([tl.reshape(-1, 1), ll.reshape(-1, 1), layer.reshape(-1, 1), sigds], axis=1)
        title = np.hstack([np.array(['stage', 'layer', 'loci']), sid])
        return merge, merge[:, 0:2], title
