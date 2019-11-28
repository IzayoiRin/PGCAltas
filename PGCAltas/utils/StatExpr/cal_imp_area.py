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

from .DataReader.reader import DataReaderBase
from .StaUtills.Pfeatures import GenericFeaturesProcess
from .const import package as c


logger = logging.getLogger("django")


class ReaderFromDimensions(DataReaderBase):

    def __init__(self, dirname, filename):
        super(ReaderFromDimensions, self).__init__(dirname, filename)
        self.samples = list()
        self.features = None
        self.labels_list_t = list()
        self.labels_list_l = list()
        self.llabels = None
        self.tlabels = None
        delattr(self, 'labels')
        delattr(self, 'labels_list')

    def workon(self, f_name, r_dframe):
        all_samples = r_dframe.columns
        pattern = re.compile(r"^\d(E?P)$")
        samples = [s for s in all_samples if re.match(pattern, s)]
        if not samples:
            return
        self.samples.extend(samples)
        locations = np.array([re.sub(pattern, lambda x: x.groups()[0], s) for s in samples])
        times = np.array([f_name.split('\\')[-1][1:4] for _ in range(len(samples))])
        df = r_dframe.loc[:, samples].T
        self.labels_list_l.append(locations)
        self.labels_list_t.append(times)
        if self.features is None:
            self.features = df.columns.to_numpy()
        return df.to_numpy()

    def get_ds_and_ls(self):
        self.dataset = np.concatenate(self.dataframes, axis=0)
        self.llabels = np.hstack(self.labels_list_l)
        self.tlabels = np.hstack(self.labels_list_t)


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


def importance_mtx(reader, dimension, split_tt=False):
    ppf = PPFeatures().init_from_data(reader.dataset, (reader.tlabels, reader.llabels))
    ppf('ONEHOT', 'IMPUTE', 'STANDARDIZE', dim=dimension)
    dataset, tlabels, llabels, features = ppf.dataset, ppf.labels[0], ppf.labels[1], reader.features
    s = SLTFeatures().init_from_data(dataset, (tlabels, llabels))
    # RDF_PARAMS in const.py
    s("RANDOM_FOREST",
      mparams=(c.RDF_PARAMS,),
      dim=dimension, split_tt=split_tt)
    s.dumps(open(os.path.join(reader.pkl_path, 'RDF%sClassifier.pkl' % dimension.title()), 'wb'))
    mtx = np.vstack([s.asc_order, features[s.asc_order], s.importance_]).T
    print('%s: Acc=%.2f' % (dimension.upper(), s.acc_))
    return mtx


def df_area(pkl):
    mtx = pickle.load(open(pkl, 'rb'))   # type: np.ndarray
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


__READER_FLUSHED = False


def execute_eim_process():
    """
    Calculate the Equ-importance Integral Matrix from dataReader built across raw_data or pkl_reader
    Output:
        .\pickles
            \OBJ*.pkl
            \RDF*Classifier.pkl & \RDF*Flow.pkl & RDF*Score.pkl
        .\texts\
            RDF*Flow.txt
    """
    if __READER_FLUSHED:
        logger.warning('Reader Flushed')
        reader = ReaderFromDimensions(c.DATA_DIR, r'.*?\.' + c.FILE_TYPE + r'$')
        reader.read(header=0, sep='\t', index_col=0).get_ds_and_ls()
        reader.dumps_as_pickle()
        logger.info('Dumps Ready')
    elif c.PKL_FILE:
        logger.info('Loading from: %s' % c.PKL_FILE)
        reader = ReaderFromDimensions.init_from_pickle(c.DATA_DIR, c.PKL_FILE)
    else:
        return

    logger.info('%s Complete Ready' % reader)
    for dim in c.DIMENSIONS:
        logger.info("Lady's Calculating Importance ...")
        mtx = importance_mtx(reader, dim, split_tt=True)
        mtx_p = os.path.join(reader.pkl_path, 'RDF%sScore.pkl' % dim.title())
        pickle.dump(mtx, open(mtx_p, 'wb'))
        logger.info("Lady's Calculating Area Curve ...")
        df = df_area(mtx_p)
        df_pp = os.path.join(reader.pkl_path, 'RDF%sFlow.pkl' % dim.title())
        df_pc = os.path.join(reader.csv_path, 'RDF%sFlow.txt' % dim.title())
        df.to_pickle(df_pp)
        df.to_csv(df_pc, sep='\t', header=True, index=False)
    logger.info('CAVED!!!')
