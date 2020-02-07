import os
import pickle
import logging

import pandas as pd

from PGCAltas.utils.StatExpr.DataReader.reader import DataReader, ReaderLoadError
from PGCAltas.utils.StatExpr.StatProcessor.FeaturesProcessor import MessProcessesError
from PGCAltas.utils.StatExpr.StatProcessor.FeaturesProcessor.processors import FeatureBasicExtractProcessor
from PGCAltas.utils.StatExpr.FunctionalDomain.temp_const import package as c


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
        # TODO: overwritten for validator: return reader with 'sigscreened' version and splitted row_no
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
        # receive two kinds of params:
        # params1: [dataset, labels] len==2
        # params2: [xtr, xte, ytr, yte] len==4
        # params1 will be departed by default test_size as shape like params2
        etp = etp_cls().init_from_data(self.get_dataset(), self.get_labels())
        return etp

    def estimate_dimension(self):
        raise NotImplementedError

    def _estimate_dimension(self, index=None, columns=None):
        dim = self.kwargs.get('dim')
        self.etp.dumps(
            open(os.path.join(self._pkl_path, '%sEstimator.pkl' % dim.title()), 'wb')
        )
        header = columns or ['D%s' % i for i in range(self.n_componets)]
        return pd.DataFrame(self.etp.dataset, index=index or self.etp.get_labels(), columns=header)

    def _dumps(self, datframe, dim, name='Estimated%sFlow'):
        p1 = os.path.join(self._csv_path, name % dim.title() + '.txt')
        datframe.to_csv(p1, sep='\t', header=True, index=True)

        p2 = os.path.join(self._pkl_path, name % dim.title() + '.pkl')
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
            # expr = self.__load__(c.SIGEXPR_PKL[k])
            expr = self.reader.dataset
            self.resolute_from_expr(expr)

            try:
                etp_df = self.estimate_dimension()
            except MessProcessesError as e:
                logging.critical(e)
                self.kwargs.get('critical', self.kwargs.setdefault('critical', [])).append('mpe')
                return
            self.estimated_dat_.append(etp_df)
            self._dumps(etp_df, k)
        # update reader
        # list: [ estimated_dat_dim1, estimated_dat_dim2, ...]
        # estimated_dat_dim: set / label / D0 / ..../ Dn
        self.reader.dataset = self.estimated_dat_
        # push to historic transformed stack
        self.reader.historic_trans['extracted'] = (self.estimated_dat_, None)
        self.reader.dumps_as_pickle(fname=self.pklfile)
