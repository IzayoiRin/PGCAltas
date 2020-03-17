import copy
import pickle

import numpy as np
import sklearn.preprocessing as pp
import sklearn.impute as ipt
import sklearn.ensemble as esb
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
# import sklearn.feature_selection as fs
from sklearn.metrics import accuracy_score

from PGCAltas.utils.statUniversal import train_test_split
from PGCAltas.utils.errors import MessProcessesError, FailInitialedError


class FeaturesProcessBase(object):

    class Meta:

        @classmethod
        def route(cls, func, callfn=None):

            def execute(self, method, *fargs, mparams=(), **fkwargs):
                name = func.__name__.split('_', 1)[1].upper()
                process = getattr(self, name, None)
                assert process, 'No such process mapping'

                method = process.get(method, None)
                assert method, 'No such scale model'
                margs, mkwargs = cls.resolute_params(mparams)
                fit = method(*margs, **mkwargs)

                if callfn:
                    fit = getattr(fit, callfn, None)
                    assert fit, 'The fit function is not callable'
                return func(self, fit, *fargs, **fkwargs)
            return execute

        @staticmethod
        def resolute_params(params):
            ret = [tuple(), dict()]
            for p in params:
                if isinstance(p, (tuple, list)):
                    ret[0] = p
                elif isinstance(p, dict):
                    ret[1] = p
            return ret

    def __init__(self):
        self.dataset = None
        self._dataset = None
        self.labels = None
        self._labels = None

    def init_from_data(self, dataset, labels):
        self.dataset = dataset
        self._dataset = copy.deepcopy(self.dataset)
        self.labels = labels
        self._labels = copy.deepcopy(self.labels)
        return self

    @classmethod
    def load(cls, buffer, **kwargs):
        return pickle.load(buffer, **kwargs)

    def dumps(self, buffer, **kwargs):
        pickle.dump(self, buffer, **kwargs)


class GenericFeaturesProcess(FeaturesProcessBase):

    meta = FeaturesProcessBase.Meta
    selector = None
    __initial = False

    def __new__(cls, *args, **kwargs):
        if not cls.__initial:
            callfn = getattr(cls, 'callfn', None)
            actions = [i for i in dir(cls) if i.startswith('fit')]
            for action in actions:
                func = getattr(cls, action)
                setattr(cls, action, cls.meta.route(func, callfn))
            cls.__initial = super().__new__(cls)
        return cls.__initial

    def __init__(self):
        super().__init__()

    def get_dataset(self, *args, **kwargs):
        return self.dataset

    def get_labels(self, *args, **kwargs):
        return self.labels

    def get_selector(self, *args, **kwargs):
        return self.selector


class WholePreProcessMixin(object):

    def fit_dimensionless(self, fit, *args, **kwargs):
        self.dataset = fit(self.dataset, self.labels, *args, **kwargs)
        return self

    def fit_binarize(self, fit, *args, **kwargs):
        self.dataset = fit(self.dataset, self.labels, *args, **kwargs)
        return self

    def fit_encode(self, fit, *args, **kwargs):
        self.labels = fit(self.labels.reshape(-1, 1), *args, **kwargs).toarray()
        return self

    def fit_na(self, fit, *args, **kwargs):
        self.dataset = fit(self.dataset, *args, **kwargs)
        return self

    def fit_funcitrans(self, fit, *args, **kwargs):
        self.dataset = fit(self.dataset, *args, **kwargs)
        return self


class FeaturesWholePreProcessor(GenericFeaturesProcess, WholePreProcessMixin):

    DIMENSIONLESS = {
        "STANDARDIZE": pp.StandardScaler,
        "MINMAX": pp.MinMaxScaler,
        "NORMALIZE": pp.Normalizer,
    }

    BINARIZE = {
        "BINARY": pp.Binarizer,
    }

    ENCODE = {
        "ONEHOT": pp.OneHotEncoder,
    }

    NA = {
        # "mean", "median", "most_frequent", "constant"
        "IMPUTE": ipt.SimpleImputer,
    }

    FUNCTRANS = {
        "POLYNOMIAL": pp.PolynomialFeatures,
        'CUSTOM': pp.FunctionTransformer,
    }


class BasicPreProcessMixin(object):

    def fit_dimensionless(self, fit, *args, **kwargs):
        labels = self.get_labels()
        self.dataset = fit(self.dataset, labels, *args, **kwargs)
        return self

    def fit_na(self, fit, *args, **kwargs):
        self.dataset = fit(self.dataset, *args, **kwargs)
        return self

    def fit_encode(self, fit, *args, **kwargs):
        self.labels = fit(self.labels.reshape(-1, 1)).toarray()
        return self


class FeaturesBasicPreProcessor(GenericFeaturesProcess, BasicPreProcessMixin):

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

    def __call__(self, method, **kwargs):
        if len(method) > 3:
            raise MessProcessesError("Wrong processes queue")
        self.kwargs = kwargs
        # fit each labels as ONE HOT CODE
        self.fit_encode(method[0], mparams=({'categories': self.kwargs.get('categories', None)}, ))
        # replace NA with most frequent value then dimensionless
        self.fit_na(method[1], mparams=({'strategy': self.kwargs.get('strategy', 'mean')}, )).\
            fit_dimensionless(method[2])


class BasicScreenMixin(object):

    def fit_ensemble(self, fit):
        xtr, ytr = self.dataset, self.get_labels()
        if self.kwargs.get('split', None):
            xtr, xte, ytr, yte = self.train_or_test()

        fit.fit(xtr, ytr)
        acc = accuracy_score(yte, fit.predict(xte)) if self.kwargs.get('split') else 1.0
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


class FeaturesBasicScreenProcessor(GenericFeaturesProcess, BasicScreenMixin):

    ENSEMBLE = {
        "RANDOM_FOREST": esb.RandomForestClassifier
    }

    spilter = {'test_size': 0.3,
               'random_state': 0}

    def train_or_test(self):
        labels = self.get_labels()
        return train_test_split(mode='L')(self.dataset, labels, **self.spilter)

    def __call__(self, method, mparams=(), **kwargs):
        # execute(self, method, *fargs, mparams=(), **fkwargs)
        # func(self, fit, *fargs, **fkwargs)
        self.kwargs = kwargs
        self.fit_ = self.fit_ensemble(method, mparams=mparams)
        self.cal_importance_rank()
        return self


class BasicExtractMixin(object):

    # def fit_reduce(self, fit, *fargs, **fkwargs):
    #     if fkwargs.get('supervised'):
    #         xtr, xte = self.dataset[self._trno, :], self.dataset[self._teno, :]
    #         labels = self.get_labels()
    #         ytr, yte = labels[self._trno], labels[self._teno]
    #         # fitting LDA model
    #         fit.fit(xtr, ytr)
    #         acc = accuracy_score(yte, fit.predict(xte))
    #         setattr(self, 'supervised_acc_', acc)
    #     else:
    #         fit.fit(self.dataset)
    #     self.dataset = fit.transform(self.dataset)
    #     return self

    def fit_reduce(self, fit, *fargs, **fkwargs):
        xtr, ytr = fargs

        if fkwargs.get('supervised'):
            # training set
            fit.fit(xtr, ytr)
            return fit

        fit.fit(xtr)
        xtr = fit.transform(xtr)
        return [xtr, ytr]


class FeatureBasicExtractProcessor(GenericFeaturesProcess, BasicExtractMixin):

    REDUCE = {
        "PRINCIPAL_COMPONENTS": PCA,
        "LINEAR_DISCRIMINANT": LinearDiscriminantAnalysis,
    }

    spilter = {
        'test_size': 0.2,
        'random_state': 0,
    }

    def __init__(self):
        super(FeatureBasicExtractProcessor, self).__init__()
        self._teno, self._trno = None, None

    def init_from_data(self, dataset, labels, training=True):
        # from testing set
        if not training:
            self.dataset, self.labels = dataset, labels
            return self

        # from training set
        if isinstance(dataset, tuple) and isinstance(labels, tuple):
            xtr, xte = dataset
            ytr, yte = labels
        elif not isinstance(dataset, tuple) and not isinstance(labels, tuple):
            self.dataset, self.labels = dataset, labels
            xtr, xte, ytr, yte = self.train_or_test()
        else:
            raise FailInitialedError('Wrong init-data format')
        # record test and train's row numbers
        self._teno, self._trno = range(ytr.shape[0], ytr.shape[0] + yte.shape[0]), range(ytr.shape[0])
        # recombination [tr, te]
        self.dataset = np.vstack([xtr, xte])
        self.labels = np.hstack([ytr, yte])
        return self

    def train_or_test(self):
        labels = self.get_labels()
        return train_test_split(mode='L')(self.dataset, labels, **self.spilter)

    # def layer_samples(self, labels):
    #     layers = np.unique(labels)
    #     ret = [np.argwhere(labels == l).reshape(-1) for l in layers]
    #     test, train = list(), list()
    #     for idxs in ret:
    #         n = idxs.shape[0]
    #         # layer with only one sample, copy itself
    #         if n == 1:
    #             idxs = np.hstack([idxs, idxs])
    #             n = 2
    #         # now, every layer with at least 2 samples, then cal testN
    #         testN = np.floor(n * self.spilter.get('test_size'))
    #         # if testN == 0, test set must have one sample, thus testN must be 1
    #         if testN == 0:
    #             testN += 1
    #         # now, every layer with at least 2 samples and its testN all larger than 0
    #         np.random.seed(self.spilter.get('random_state'))
    #         # random select from WHOLE SET idxs
    #         testingSet = np.random.choice(idxs, size=int(testN))
    #         test.append(testingSet)
    #         # the diff set between WHOLE SET idxs and SUB SET testingSet
    #         trainingSet = np.setdiff1d(idxs, testingSet)
    #         train.append(trainingSet)
    #
    #     test, train = np.hstack(test), np.hstack(train)
    #     return self.dataset[train, :], self.dataset[test, :], labels[train], labels[test]

    def __call__(self, method, mparams=(), **kwargs):
        self.kwargs = kwargs
        self.fit_reduce(method, mparams=mparams, supervised=True)


class FeatureFilterExtractProcessor(FeatureBasicExtractProcessor):

    def __call__(self, *methods, **kwargs):
        """
        init_from_data:
        train --1:
            self.dataset, self.labels ----> whole
            self._teno, self._trno ----> rows
        train --0:
            self.dataset, self.labels ----> vd
            self._teno, self._trno ----> None

        __call__:

        train --1:
            fit_reduce(tr) ----> fit
        train --0:
            load_fit() ----> fit
        fit ----> predict(te)

        :param methods: [(filter, flt_params), (reducer, rdc_params)]
        :param kwargs:
        :return:
        """
        self.kwargs = kwargs
        if len(methods) != 2:
            raise MessProcessesError("Wrong processes queue")
        (flt, flt_params), (rdc, rdc_params) = methods
        ori_features = self.dataset.shape[1]
        flag = ori_features > flt_params['n_components'] > rdc_params['n_components'] > 0
        if not flag:
            raise MessProcessesError(
                "Wrong processes params: features[%d], filter[%d], components[%d]" %
                (ori_features, flt_params['n_components'], rdc_params['n_components'])
            )

        def _fitting(xtr, xte, ytr, yte):
            tr_set = xtr, ytr
            te_set = xte, yte

            # PCA reduce training set, return PCA reduced data
            tr_re = self.fit_reduce(flt, *tr_set, mparams=(flt_params,), supervised=False)  # type: tuple
            # LDA fitting training set, return fitted model
            fit = self.fit_reduce(rdc, *tr_re, mparams=(rdc_params,), supervised=True)

            # PCA reduce testing set and LDA predicting testing set, return PCA reduced data
            te_re = _predict(fit, *te_set)
            # LDA reduce whole set
            self.dataset = fit.transform(np.vstack([tr_re[0], te_re[0]]))
            return fit

        def _predict(fit, *te_set):
            """PCA reduce testing set and LDA predicting testing set, return PCA reduced data"""
            re = self.fit_reduce(flt, *te_set, mparams=(flt_params,), supervised=False)  # type: tuple
            ypr = fit.predict(re[0])
            acc = accuracy_score(re[1], ypr)
            setattr(self, 'supervised_acc_', acc)
            return re

        # training model, return fitted model
        if kwargs.get('training', True):
            xtr, xte = self.dataset[self._trno, :], self.dataset[self._teno, :]
            labels = self.get_labels()
            ytr, yte = labels[self._trno], labels[self._teno]
            return _fitting(xtr, xte, ytr, yte)

        # testing model, return None
        else:
            fit = kwargs.get('loaded_fit')
            if fit:
                re = _predict(fit, self.dataset, self.get_labels())
                self.dataset = fit.transform(re[0])


class Viewer2DMixin(object):

    def fit_estimate(self, fit, *fargs, **fkwargs):
        self.dataset = fit.fit_transform(self.dataset)
        return fit


class Feature2DViewerProcessor(GenericFeaturesProcess, Viewer2DMixin):

    ESTIMATE = {
        "T_STOCHASTIC": TSNE,
        "SPARSE_SVD": TruncatedSVD
    }

    def __call__(self, method, mparams=(), **kwargs):
        self.kwargs = kwargs
        self.fit_ = self.fit_estimate(method, mparams=mparams)
