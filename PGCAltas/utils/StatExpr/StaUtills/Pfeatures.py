import copy
import pickle

from sklearn.datasets import load_iris


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


class PreProcessMixin(object):

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


class FeaturesSelectMixin(object):

    def fit_filter(self, fit, *args, **kwargs):
        self.dataset = fit(self.dataset, *args, **kwargs)
        return self


class MyFeatures(GenericFeaturesProcess, PreProcessMixin):

    import sklearn.preprocessing as pp
    import sklearn.impute as ipt
    import sklearn.feature_selection as fs
    import numpy as np
    from scipy.stats import pearsonr

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

    FILTER = {
        "VARIANCE": fs.VarianceThreshold,
        "PEARSONR": {
            'PEARSONR': fs.SelectKBest,
            'score_func': lambda X, y: list(np.array(list(map(lambda x: pearsonr(x, y), X.T))).T)
        },
        "CHI2": {
            'func': fs.SelectKBest,
            'margs': []
        }
    }


def main():
    import numpy as np
    iris = load_iris()
    label_names = iris.target_names
    feature_names = iris.feature_names
    dataset = iris.data
    labels = iris.target
    fp = MyFeatures().init_from_data(dataset, labels)


if __name__ == '__main__':
    main()
