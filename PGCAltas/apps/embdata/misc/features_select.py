import numpy as np
import pandas as pd

from PGCAltas.utils.StatExpr.cal_imp_area import GenericEIMProcess
from embdata.misc.reader import BinomialDataReader


class EMBTABinomalEIMProcess(GenericEIMProcess):

    data_reader_class = BinomialDataReader

    def __init__(self, filename):
        super().__init__(filename)

    def importance_mtx(self, *args, **kwargs):
        self.ppf = self.get_preprocessor()
        self.ppf(*self.preprocesses)
