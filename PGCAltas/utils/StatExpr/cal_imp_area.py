import logging
import re
import os

import numpy as np
import pandas as pd
import pickle

import sklearn.preprocessing as pp
import sklearn.impute as ipt
import sklearn.ensemble as esb
import sklearn.feature_selection as fs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from PGCAltas.utils.StatExpr.DataReader.reader import DataReader, ReaderLoadError
from .StaUtills.Pfeatures import GenericFeaturesProcess
from .const import package as c


logger = logging.getLogger("django")


class PerProcessedMixin(object):

    def fit_dimensionless(self, fit, label_dim, *args, **kwargs):
        labels = self.get_labels(label_dim)
        self.dataset = fit(self.dataset, labels, *args, **kwargs)
        return self

    def fit_na(self, fit, *args, **kwargs):
        self.dataset = fit(self.dataset, *args, **kwargs)
        return self

    def fit_encode(self, fit, *args, **kwargs):
        self.labels = [fit(ls.reshape(-1, 1), *args, **kwargs).toarray()
                       for ls in self.labels]
        return self


class PPFeatures(GenericFeaturesProcess, PerProcessedMixin):

    callfn = "fit_transform"

    DIMENSIONLESS = {
        "STANDARDIZE": pp.StandardScaler,
        "MINMAX": pp.MinMaxScaler,
        "NORMALIZE": pp.Normalizer
    }

    NA = {
        # "mean", "median", "most_frequent", "constant"
        "IMPUTE": ipt.SimpleImputer,
    }

    ENCODE = {
        "ONEHOT": pp.OneHotEncoder,
    }

    def get_labels(self, label_dim):
        label_dim = 0 if label_dim == 'time' else 1
        return self.labels[label_dim]

    def __call__(self, *method, dim=None):
        # fit each labels as ONE HOT CODE
        self.fit_encode(method[0])
        # replace NA with most frequent value then dimensionless
        self.fit_na(method[1], mparams=({'strategy': 'mean'},)).\
            fit_dimensionless(method[2], dim)


class FeaturesScreenMixin(object):

    def fit_ensemble(self, fit, dim, split_tt):
        xtr, ytr = self.dataset, self.get_labels(dim)
        if split_tt:
            xtr, xte, ytr, yte = self.train_or_test(dim)
        fit.fit(xtr, ytr)
        acc = accuracy_score(yte, fit.predict(xte)) if split_tt else 1.0
        setattr(self, 'acc_', acc)
        return fit

    def cal_importance_rank(self):
        fit = getattr(self, 'fit_', None)
        if fit is None:
            return
        m, n = self.dataset.shape
        improtances = fit.feature_importances_ * n   # type: np.ndarray
        self.asc_order = improtances.argsort()
        self.importance_ = improtances[self.asc_order]


class SLTFeatures(GenericFeaturesProcess, FeaturesScreenMixin):

    ENSEMBLE = {
        "RANDOM_FOREST": esb.RandomForestClassifier
    }

    spilter = {'test_size': 0.3,
               'random_state': 0}

    selector = {
        'RANDOM_FOREST': fs.SelectFromModel
    }

    def get_labels(self, label_dim):
        label_dim = 0 if label_dim == 'time' else 1
        return self.labels[label_dim]

    def train_or_test(self, dim):
        labels = self.get_labels(dim)
        dataset = self.get_dataset()
        return train_test_split(dataset, labels, **self.spilter)

    def get_selector(self, method):
        selector = self.selector.get(method, None)
        return selector

    def __call__(self, method, mparams=(), dim=None, split_tt=False):
        # execute(self, method, *fargs, mparams=(), **fkwargs)
        # func(self, fit, *fargs, **fkwargs)
        self.fit_ = self.fit_ensemble(method, dim, split_tt, mparams=mparams)
        self.cal_importance_rank()
        return self


class EIMProcess(object):

    __READER_FLUSHED = False

    data_reader_class = DataReader

    preprocessor_class = PPFeatures
    preprocesses = ['ONEHOT', 'IMPUTE', 'STANDARDIZE']

    select_processor_class = SLTFeatures
    select_process = "RANDOM_FOREST"
    select_process_params = (c.RDF_PARAMS,)

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
            reader.dumps_as_pickle()
            logger.info('Dumps Ready')
        elif self.pklfile:
            logger.info('Loading from: %s' % self.pklfile)
            reader = self.data_reader_class.init_from_pickle(self.dirname, self.pklfile)

        return reader

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
        return self.select_processor_class

    def get_select_processor(self):
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

    def execute_eim_process(self):
        """
        Calculate the Equ-importance Integral Matrix from dataReader built across raw_data or pkl_reader
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

        logger.info('%s Complete Ready' % self.reader)

        for dim in self.dimensions:
            logger.info("Lady's Calculating Importance ...")

            mtx = self.importance_mtx(dim, split_tt=True)
            mtx_p = os.path.join(self.reader.pkl_path, 'RDF%sScore.pkl' % dim.title())
            pickle.dump(mtx, open(mtx_p, 'wb'))

            logger.info("Lady's Calculating Area Curve ...")

            df = self.df_area(mtx_p)
            df_pp = os.path.join(self.reader.pkl_path, 'RDF%sFlow.pkl' % dim.title())
            df_pc = os.path.join(self.reader.csv_path, 'RDF%sFlow.txt' % dim.title())
            df.to_pickle(df_pp)
            df.to_csv(df_pc, sep='\t', header=True, index=False)

        logger.info('CAVED!!!')
