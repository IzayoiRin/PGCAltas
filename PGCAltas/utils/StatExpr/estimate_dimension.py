import os
import pickle
import logging

import pandas as pd

from PGCAltas.utils.StatExpr.DataReader.reader import DataReader, ReaderLoadError
from PGCAltas.utils.StatExpr.StaUtills.FeaturesProcessor import MessProcessesError
from PGCAltas.utils.StatExpr.StaUtills.FeaturesProcessor.processors import FeatureBasicExtractProcessor
from .const import package as c


logger = logging.getLogger("django")


class DimensionEstimate(object):

    data_reader_class = DataReader
    estimate_processor_class = FeatureBasicExtractProcessor
    estimate_process = "LINEAR_DISCRIMINANT"
    estimate_process_params = c.LDA_PARAMS,
    dimension = list()

    def __init__(self, dirname=None, pklfile=None):
        self.dirname = dirname or c.DATA_DIR
        self.pklfile = pklfile or c.PKL_FILE

        self.reader = None
        self._pkl_path, self._csv_path = None, None

        self.dataset = None
        self.labels = None

        self.etp = None
        self.estimated_dat_ = list()

    def get_data_reader_class(self):
        return self.data_reader_class

    def get_data_reader(self):
        dr_cls = self.get_data_reader_class()
        reader = dr_cls.init_from_pickle(self.dirname, self.pklfile)
        return reader

    def get_file_path(self):
        pkl_path, csv_path = self.reader.pkl_path, self.reader.csv_path
        return pkl_path, csv_path

    def __load__(self, v):
        p = os.path.join(self._pkl_path, v)
        with open(p, 'rb') as f:
            val = pickle.load(f)
        return val

    def resolute_from_expr(self, expr):
        return expr

    def get_labels(self):
        return self.labels

    def get_dataset(self):
        return self.dataset

    def get_estimate_processor_class(self):
        return self.estimate_processor_class

    def get_estimate_processor(self):
        etp_cls = self.get_estimate_processor_class()
        etp = etp_cls().init_from_data(self.get_dataset(), self.get_labels())
        return etp

    def estimate_dimension(self):
        raise NotImplementedError

    def _estimate_dimension(self):
        dim = self.kwargs.get('dim')
        self.etp.dumps(
            open(os.path.join(self._pkl_path, '%sEstimator.pkl' % dim.title()), 'wb')
        )
        header = ['D%s' % i for i in range(self.n_componets)]
        return pd.DataFrame(self.etp.dataset, index=self.get_labels(), columns=header)

    def _dumps(self, datframe, dim, name='Estimated%sFlow.txt'):
        p1 = os.path.join(self._csv_path, name % dim.title())
        datframe.to_csv(p1, sep='\t', header=True, index=True)

        p2 = os.path.join(self._pkl_path, name % dim.title())
        datframe.to_pickle(p2)

    def execute_estimate_process(self, **kwargs):
        self.reader = self.get_data_reader()
        if self.reader is None:
            raise ReaderLoadError("Can't load dataReader: %s" % self.data_reader_class.__name__)
        self._pkl_path, self._csv_path = self.get_file_path()

        logger.info('Load dataReader: %s' % self.reader)

        self.kwargs = kwargs
        self.n_componets = kwargs.get('n_components', None)

        for k in self.dimension:
            logger.info("Lady's Estimating Dimension ...")

            self.kwargs['dim'] = k
            expr = self.__load__(c.SIGEXPR_PKL[k])
            self.resolute_from_expr(expr)

            try:
                etp_df = self.estimate_dimension()
            except MessProcessesError as e:
                logging.critical(e)
                return
            self.estimated_dat_.append(etp_df)
            self._dumps(etp_df, k)
