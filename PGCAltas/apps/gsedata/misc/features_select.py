import numpy as np

from PGCAltas.utils.StatExpr.FunctionalDomain.EIMPCalculator import GenericEIMProcess
from PGCAltas.utils.StatExpr.FunctionalDomain.EIMPAnalyzer import EIMAnalysis
from gsedata.misc.features_processor import GSEBinDimensionPPFeatures, GSEBinDimensionSLTFeatures
from gsedata.misc.reader import ReaderFromDimensions


class GSEBinDimensionEIMProcess(GenericEIMProcess):

    data_reader_class = ReaderFromDimensions
    preprocessor_class = GSEBinDimensionPPFeatures
    screen_processor_class = GSEBinDimensionSLTFeatures

    def __init__(self, filename):
        super().__init__(filename)

    def get_preprocessor(self):
        ppf_cls = self.preprocessor_class()
        ppf = ppf_cls.init_from_data(self.reader.dataset, (self.reader.tlabels, self.reader.llabels))
        return ppf

    def get_screen_processor(self):
        slp_cls = self.get_select_processor_class()
        dataset, tlabels, llabels = self.ppf.dataset, self.ppf.labels[0], self.ppf.labels[1],
        slp = slp_cls().init_from_data(dataset, (tlabels, llabels))
        return slp

    def importance_mtx(self, dimension, split_tt=False):
        self.ppf = self.get_preprocessor()
        self.ppf(self.preprocesses, dim=dimension)
        features = self.reader.features
        self.slp = self.get_screen_processor()
        # RDF_PARAMS in temp_const.py
        self.slp(self.screen_process,
                 mparams=self.screen_process_params,
                 dim=dimension, split=split_tt)

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
