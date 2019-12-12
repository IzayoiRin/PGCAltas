import pickle

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class ModelProcessError(Exception):
    pass


class CannotMoveError(Exception):
    pass


class EnsembleFit(object):

    def __init__(self, alpha, g):
        self.alpha_ = alpha.reshape(-1, 1)
        self.g_ = g
        self.yprob_ = None

    def predict(self, x):
        if len(self.alpha_):
            # label predict
            pre = np.array([g.predict(x) for g in self.g_])
            esm_pre = np.sign(self.alpha_.T.dot(pre)).reshape(-1)
            # probability predict
            pre_prob = np.array([g.decision_function(x) for g in self.g_])
            esm_prob = self.alpha_.T.dot(pre_prob)

            self.yprob_ = esm_prob.reshape(-1)
            return esm_pre
        else:
            g = self.g_[0]
            self.yprob_ = g.decision_function(x)
            return g.predict(x)

    def score_(self, y, ypre):
        return np.mean(ypre == y)

    def roc_(self, y, ypre, **kwargs):
        keys = ['tpr', 'fpr', 'threshold']
        return dict(zip(keys, roc_curve(y, ypre, **kwargs)))

    def auc_(self, y, ypre, **kwargs):
        return roc_auc_score(y, ypre, **kwargs)

    def ap_(self, y, ypre, **kwargs):
        return average_precision_score(y, ypre, **kwargs)


class AdaBoost(object):

    def __init__(self, n_boost=10):
        self.weights = None
        self.instance = None
        self.n_boost = n_boost

        self.alpha_arr = None
        self.g_arr = None

    def _init_weight(self):
        m, n = self.instance.xtr.shape
        self.weights = np.ones(m) / m

        self.alpha_arr = list()
        self.g_arr = list()

    def _boost(self, g):
        ypre = g.predict(self.instance.xtr)
        # Em mask vector
        mask_vec = np.array([int(i) for i in ypre != self.instance.ytr])
        # Em = sum(wi * I(g(x) != yi))
        er = np.sum(mask_vec * self.weights)
        if er == 0:
            raise CannotMoveError
        # iter coefficient
        alpha = 1 / 2 * np.log((1 - er) / er)
        self.alpha_arr.append(alpha)
        self.update_weights(alpha, ypre)

    def update_weights(self, alpha, ypre):
        # update weights
        u = np.exp(-alpha * ypre * self.instance.ytr)
        normal_coe = self.weights.T.dot(u)
        self.weights *= u / normal_coe

    def boost(self, n_boost=None):
        func = None
        self.n_boost = n_boost or self.n_boost

        def inner(f):
            nonlocal func
            func = f
            return calling

        def calling(ins):
            self.instance = ins
            self._init_weight()
            n_boost = getattr(ins, 'n_boost', self.n_boost)
            for i in range(n_boost):
                # calculate Gm(x)
                g = func(ins, sample_weight=self.weights * ins.ytr.shape[0])
                try:
                    self._boost(g)
                # boost can't move
                except CannotMoveError:
                    # alpha not existed == boost can't move at first iter
                    if len(self.alpha_arr) == 0:
                        # only one Gm appended to g_arr
                        self.g_arr.append(g)
                    # alpha existed == boost can't move at m iter
                    # len(alpha_arr) == len(g_arr) - 1, Gm should not be appended to g_arr
                    ins.err_flag_ = "Boost Can't Move"
                # boost can move, Gm appended to g_arr
                else:
                    self.g_arr.append(g)
            self.alpha_arr, self.g_arr = np.array(self.alpha_arr), np.array(self.g_arr)
            fit = EnsembleFit(self.alpha_arr, self.g_arr)
            ins.fit = fit
            return fit

        return inner


ada = AdaBoost()


class GenericModelClassifier(object):

    model_class = None
    test_size = 0.6

    def __init__(self):
        self.dataset = None
        self.labels = None
        self.fit = None
        self.__split = False

        self.ypre_ = None
        self.ytru_ = None

    def initial_from_data(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels
        self._split_test_or_train()

    def get_model(self):
        return self.model_class()

    def _split_test_or_train(self):
        if self.__split:
            return
        self.__split = True
        self.xtr, self.xte, self.ytr, self.yte = train_test_split(self.dataset, self.labels,
                                                                  test_size=self.test_size,
                                                                  random_state=0)

    def fit_model(self, sample_weight=None):
        model = self.get_model()
        if model is None:
            raise ModelProcessError('No model class, set clsAttr: model_class first')
        clf = model.fit(self.xtr, self.ytr, sample_weight)
        self.fit = clf
        return clf

    def predict(self):
        ypre = self.fit.predict(self.xte)
        self.ypre_ = ypre
        return ypre

    def _analysis_pre(self, fn, *refits, **kwargs):
        raise NotImplementedError

    @classmethod
    def load(cls, buffer, **kwargs):
        return pickle.load(buffer, **kwargs)

    def dumps(self, buffer, **kwargs):
        pickle.dump(self, buffer, **kwargs)


class AnalysisMixin(object):

    def score(self, *refits):
        return self._analysis_pre('score_', *refits)

    def roc(self, *refits, prob=True):
        return self._analysis_pre('roc_', *refits, prob=prob)

    def auc(self, *refits, prob=True):
        return self._analysis_pre('auc_', *refits, prob=prob)

    def ap(self, *refits, prob=True):
        return self._analysis_pre('ap_', *refits, prob=prob)


class SupportVectorMachineClassifier(GenericModelClassifier, AnalysisMixin):

    model_class = SVC

    @ada.boost()
    def fit_model(self, sample_weight=None):
        return super().fit_model(sample_weight)

    def _analysis_pre(self, fn, *refits, **kwargs):
        if self.ytru_ is None:
            self.ytru_ = self.yte

        if refits:
            x, self.ytru_ = refits
            self.ypre_ = self.fit.predict(x)

        if self.ypre_ is None:
            raise ModelProcessError("Non predict y value, use method: .predict first")
        if kwargs.pop("prob", None):
            ypre = self.fit.yprob_
        else:
            ypre = self.ypre_
        fn = getattr(self.fit, fn, None)
        if fn is None:
            raise AttributeError
        return fn(self.ytru_, ypre, **kwargs)
