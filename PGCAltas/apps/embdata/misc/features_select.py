import numpy as np
import pandas as pd

from PGCAltas.utils.StatExpr.cal_imp_area import GenericEIMProcess
from embdata.misc.reader import BinomialDataReader
from ..data_const import POS, NEG, FILE_TYPE


class EMBTABinomalEIMProcess(GenericEIMProcess):

    data_reader_class = BinomialDataReader

    def __init__(self, filename):
        super().__init__(filename, pos=POS + FILE_TYPE, neg=NEG + FILE_TYPE)

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
