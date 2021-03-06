import logging
import os

import numpy as np
import pandas as pd
import pickle

from PGCAltas.utils.StatExpr.DataReader.reader import DataReader
from PGCAltas.utils.errors import ReaderLoadError
from PGCAltas.utils.StatExpr.StatProcessor.FeaturesProcessor.processors import FeaturesBasicPreProcessor, \
    FeaturesBasicScreenProcessor

from PGCAltas.utils.StatExpr.FunctionalDomain.temp_const import package as c


logger = logging.getLogger("django")


class GenericEIMProcess(object):

    __READER_FLUSHED = False

    data_reader_class = DataReader

    preprocessor_class = FeaturesBasicPreProcessor
    preprocesses = ['ONEHOT', 'IMPUTE', 'STANDARDIZE']

    screen_processor_class = FeaturesBasicScreenProcessor
    screen_process = "RANDOM_FOREST"
    screen_process_params = [c.RDF_PARAMS, ]

    def __init__(self, filename, dirname=None, pklfile=None, dims=None, **rdparams):
        self.filename = filename + c.FILE_TYPE + r'$'
        self.dirname = dirname or c.DATA_DIR
        self.pklfile = pklfile or c.PKL_FILE
        self.rdparams = rdparams

        self.dimensions = dims or c.DIMENSIONS

        self.reader = None
        self.ppf = None
        self.slp = None

    def build_data_reader(self):
        reader = None
        if self.__READER_FLUSHED:
            logger.warning('Reader Flushed')
            reader = self.data_reader_class(self.dirname, self.filename, **self.rdparams)
            reader.read(header=0, sep='\t', index_col=0).get_ds_and_ls()
            # flush the pkl_file address in memory cache
            reader.dumps_as_pickle()
            self.pklfile = c.PKL_FILE = reader.pklname
            logger.info('Dumps Ready')
        elif self.pklfile:
            logger.info('Loading from: %s' % self.pklfile)
            reader = self.data_reader_class.init_from_pickle(self.dirname, self.pklfile)
        return reader

    def pre_process(self):
        raise NotImplementedError

    def importance_mtx(self, *args, **kwargs):
        raise NotImplementedError

    def _imp_mtx_processing(self, features, dim):
        self.slp.dumps(
            open(os.path.join(self.reader.pkl_path, 'RDF%sClassifier.pkl' % dim.title()), 'wb')
        )
        mtx = np.vstack([self.slp.asc_order, features[self.slp.asc_order], self.slp.importance_]).T
        print('%s: Acc=%.2f' % (dim.upper(), self.slp.acc_))
        return mtx

    def get_preprocessor_class(self):
        return self.preprocessor_class

    def get_preprocessor(self):
        ppf_cls = self.get_preprocessor_class()
        ppf = ppf_cls().init_from_data(self.reader.dataset, self.reader.labels)
        return ppf

    def get_select_processor_class(self):
        return self.screen_processor_class

    def get_screen_processor(self):
        slp_cls = self.get_select_processor_class()
        slp = slp_cls().init_from_data(self.ppf.dataset, self.ppf.labels)
        return slp

    @staticmethod
    def df_area(pkl):
        mtx = pickle.load(open(pkl, 'rb'))  # type: np.ndarray
        s = mtx[:, 2]
        ret = list()
        sum_area = 0.0
        for i in s:
            sum_area += i
            ret.append(sum_area)
        sum_area = np.array(ret, dtype=np.float64).reshape(-1, 1)
        mtx = np.hstack([mtx, sum_area])
        df = pd.DataFrame(mtx, columns=['IDX', 'GENE', 'IMP', 'AREA'])
        print("Matrix: R[%s * %s]" % mtx.shape)
        return df

    def flushed(self):
        self.__READER_FLUSHED = True

    def execute_eim_process(self, train=True):
        """
        Calculate the Equ-importance Integral Matrix from dataReader built across raw_data or pkl_reader

        param -train controlled whether doing the whole Calculating flow,
        however, param -flush controlled whether rollback the data's version

        train -- 0 flush -- 1 Only for NEW Testing data
        not redefine Selector but flushed data from raw file, thus must be preprocessed
        NOT Necessary to rollback data's version [unitest passed]

        train -- 0 flush -- 0 A way passing this stage
        nether redefine Selector nor flushed data, thus must be preprocessed
        Necessary to rollback data's version [unitest passed]

        train -- 1 flush -- 0 Used for redefined RDF Selector from EXISTED data
        redefine Selector but not flushed data, thus must be preprocessed and fitting RDF score
        Necessary to rollback data's version [unitest passed]

        train -- 1 flush -- 1 Only for NEW Training data
        redefine Selector and flushed data, thus must be preprocessed and fitting RDF score
        NOT Necessary to rollback data's version [unitest passed]

        Output:
            .\pickles
                \OBJ*.pkl
                \RDF*Classifier.pkl & \RDF*Flow.pkl & RDF*Score.pkl
            .\texts\
                RDF*Flow.txt
        """
        self.reader = self.build_data_reader()
        if self.reader is None:
            raise ReaderLoadError("Can't load dataReader: %s" % self.data_reader_class.__name__)

        # reader's data rollback to original version
        if not self.__READER_FLUSHED:
            self.reader.dataset, self.reader.labels = self.reader.historic_trans['original']
            self.reader.historic_trans = {"original": (self.reader.dataset, self.reader.labels)}

        logger.info('%s Complete Ready' % self.reader)

        # Not redefine Selector
        if not train:
            logger.info("Lady's Preprocessing Data ...")
            self.pre_process()
            return

        # Redefine Selector must be done with repre-processing
        for dim in self.dimensions:

            logger.info("Lady's Calculating Importance ...")

            try:
                mtx = self.importance_mtx(dim, split_tt=True)
            except Exception as e:
                logging.critical(e)
                return
            mtx_p = os.path.join(self.reader.pkl_path, 'RDF%sScore.pkl' % dim.title())
            pickle.dump(mtx, open(mtx_p, 'wb'))

            logger.info("Lady's Calculating Area Curve ...")

            df = self.df_area(mtx_p)
            df_pp = os.path.join(self.reader.pkl_path, 'RDF%sFlow.pkl' % dim.title())
            df_pc = os.path.join(self.reader.csv_path, 'RDF%sFlow.txt' % dim.title())
            df.to_pickle(df_pp)
            df.to_csv(df_pc, sep='\t', header=True, index=False)

        logger.info('CAVED!!!')
