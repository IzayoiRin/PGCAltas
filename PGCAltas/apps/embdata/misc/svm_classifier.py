import logging
import os
import pickle
import time

from tqdm import tqdm

import numpy as np
import pandas as pd

from PGCAltas.utils.StatExpr.StatProcessor.ClassiferProcessor.trad_classifier import SupportVectorMachineClassifier
from PGCAltas.utils.errors import ReaderLoadError
from embdata.data_const import DATA_DIR, PKL_FILE, SVM_PARAMS, ENSEMBLE_SVM_CLASSIFIER_PKL
from embdata.misc.reader import BinomialDataReader


logger = logging.getLogger("django")


class EMBTASVMClassifier(SupportVectorMachineClassifier):

    def __init__(self):
        super().__init__()

        self.roc_mtxs = list()
        self.auc_arr = list()
        self.acc_arr = list()
        self.acc_mtx = list()
        self.fits = list()

        self.records = None

    def get_model(self):
        return self.model_class(**SVM_PARAMS)

    def acc_bar(self, y):
        acc_m = np.equal(self.ypre_, y).astype(np.int8)
        acc = np.mean(acc_m)
        acc_m = np.vstack([y, acc_m])
        return acc_m, acc  # (ndarray.shape(2, n), scale)

    def __boost(self, n):
        self.n_boost = n
        # fit model from training set
        self.fit_model()
        self.predict()
        acc_m, acc = self.acc_bar(self.yte)
        self.acc_arr.append(acc)
        self.acc_mtx.append(acc_m)

        roc, auc = self.roc(), self.auc()

        self.roc_mtxs.append(roc)
        self.auc_arr.append(auc)
        self.fits.append(self.fit)

        return acc, auc

    def adaboost(self, n_estimator: int, record_freq=None):
        if record_freq is None:
            self.records = [n_estimator]
            self.__boost(n_estimator)
            delattr(self, 'fits')
            return

        self.records = list(range(1, n_estimator + 1, record_freq))
        if n_estimator % record_freq != 1 and record_freq > 1:
            self.records.append(n_estimator)

        bar = tqdm(enumerate(self.records), total=len(self.records))
        for idx, i in bar:
            acc, auc = self.__boost(i)
            bar.set_description_str("BoostBatch_%s: N=%s, Acc=%.6f, AUC=%.6f" % (idx, i, acc, auc))
            if hasattr(self, "err_flag_"):
                time.sleep(0.1)
                print("\nBoostBatch_%s: %s" % (idx, self.err_flag_))
                self.records = self.records[:idx+1]
                break

    def roc_auc(self):
        """
        self.roc_mtxs: [roc_dict, ...., ]
        self.auc_arr: [auc, ..., ..., ]
        :return:
        """
        ret = list()
        for idx, n in enumerate(self.records):
            d = self.roc_mtxs[idx]
            auc = self.auc_arr[idx]

            temp = [
                np.array([n for _ in d['threshold']]),
                np.array([auc for _ in d['threshold']]),
                d['fpr'],
                d['tpr'],
            ]

            ret.append(np.vstack(temp))
        # ret = [4*N, ...]
        sta_mtx = np.hstack(ret).T
        df = pd.DataFrame(sta_mtx, columns=["Boost", "AUC", "FPR", "TPR"])
        return df

    def accuracy(self):
        ret = list()
        for idx, n in enumerate(self.records):
            acc = self.acc_arr[idx]  # [scale, ..., scale]
            acc_m = self.acc_mtx[idx]  # [ndarray.shape(2, n), ]
            length = range(acc_m.shape[-1])
            temp = [
                np.array([n for _ in length]),
                np.array([acc for _ in length]),
                np.array([i + 1 for i in length]),
                acc_m,
            ]
            ret.append(np.vstack(temp))
        # ret = [5*N, ...] --- hstack = shape(5, N*n) --- .T = shape(N*n, 5)
        sta_mtx = np.hstack(ret).T
        df = pd.DataFrame(sta_mtx, columns=["Boost", "Acc", "Case", "Label", "Predict"])
        return df


class EMBTASVMProcess(object):

    data_reader_class = BinomialDataReader
    classifier_class = EMBTASVMClassifier
    test_size = 0.4
    dimension = ['binomial', ]

    def __init__(self, dirname=None, pklfile=None):
        self.dirname = dirname or DATA_DIR
        self.pklfile = pklfile or PKL_FILE

        self.reader = None
        self._pkl_path, self._csv_path = None, None

        self.dataset = None
        self.labels = None
        self.sample_types = None

        self.primary_dim = None

    def get_data_reader_class(self):
        return self.data_reader_class

    def get_data_reader(self):
        if self.reader:
            return self.reader

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
        """
        training -- 1(training model) or -1(validating model):
           self.dataset, self.labels, self.sample_types : (tr, te)

        training training -- 0(predicting model):
           self.dataset, self.labels, self.sample_types : arrays
        :param expr: historic_trans[''extracted'']
        """
        self.dataset = expr.iloc[:, 2:].to_numpy(dtype=np.float32)  # type: np.ndarray
        self.labels = expr.loc[:, 'label'].to_numpy(dtype=np.int8)  # type: np.ndarray
        self.sample_types = expr.index.to_numpy(dtype=np.str)  # type: np.ndarray

        self.primary_dim = self.dataset.shape[1]

        if self.kwargs['training'] in [1, -1]:
            self.dataset = self.dataset[self.reader.tr_rows], self.dataset[self.reader.te_rows]
            self.labels = self.labels[self.reader.tr_rows], self.labels[self.reader.te_rows]
            self.sample_types = self.sample_types[self.reader.tr_rows], self.sample_types[self.reader.te_rows]

    def get_classifier_class(self):
        return self.classifier_class

    def get_classifier(self):
        # classifier from testing set
        if self.kwargs['training'] is 0:
            obj = self.__load__(ENSEMBLE_SVM_CLASSIFIER_PKL[self.kwargs['dim']])
            obj.initial_from_data(self.get_dataset(), self.get_labels(), training=False)
            if hasattr(obj, 'fits'):
                obj.fit = obj.fits[self.kwargs.get('best_n', -1)]
        # classifier from training set
        else:
            cls = self.get_classifier_class()
            obj = cls()
            obj.initial_from_data(self.get_dataset(), self.get_labels())
        return obj

    def get_labels(self):
        return self.labels

    def get_dataset(self):
        return self.dataset

    def _dumps(self, datframe, name):
        dim = self.kwargs['dim']
        p1 = os.path.join(self._csv_path, name % dim.title() + '.txt')
        datframe.to_csv(p1, sep='\t', header=True, index=True)

        p2 = os.path.join(self._pkl_path, name % dim.title() + '.pkl')
        datframe.to_pickle(p2)

    def predicted_expr(self, ypr):
        """get predicted samples' expression matrix, support Costume Overwrite"""
        posid = np.argwhere(ypr == 1).reshape(-1)
        negid = np.argwhere(ypr == 0).reshape(-1)
        negid = np.random.choice(negid, size=posid.shape[0], replace=False)
        sigexpr_pos = self.reader.historic_trans['sigscreened'][0].iloc[posid, :]
        sigexpr_neg = self.reader.historic_trans['sigscreened'][0].iloc[negid, :]
        self._dumps(sigexpr_pos, name="SVMClassifier%sFlowPredictExprPos")
        self._dumps(sigexpr_neg, name="SVMClassifier%sFlowPredictExprNeg")

    def execute_classify_process(self, callback=None, **kwargs):
        self.reader = self.get_data_reader()
        if self.reader is None:
            raise ReaderLoadError("Can't load dataReader: %s" % self.data_reader_class.__name__)
        self._pkl_path, self._csv_path = self.get_file_path()

        logger.info('Load dataReader: %s' % self.reader)

        self.kwargs = kwargs
        self.kwargs['training'] = self.kwargs.get('training', 1)
        n_estimator = kwargs.pop("n_estimator", 10)
        rec_frequency = kwargs.pop("record_freq", None)

        for idx, k in enumerate(self.dimension):
            logger.info("Lady's Simulating Dimension ...")

            self.kwargs['dim'] = k
            expr = self.reader.dataset[idx]
            self.resolute_from_expr(expr)

            print('Process Model: %d [in-factors: %d]' % (self.kwargs['training'], self.primary_dim))

            # init svm classifier
            svm = self.get_classifier()
            if self.kwargs['training'] is 0:
                ypr = svm.predict().astype(np.int8)
                predict_df = pd.DataFrame(np.vstack([self.reader.samples, ypr]).T,
                                          columns=['Samples', 'PreLabels'])
                self._dumps(predict_df, name="SVMClassifier%sFlowPredict")
                self.predicted_expr(ypr)
                return

            if self.kwargs['training'] in [1, -1]:
                print("Training size / Testing size : %d / %d [%.4f]" %
                      (svm.ytr.shape[0], svm.yte.shape[0], svm.ytr.shape[0] / svm.yte.shape[0])
                      )
                time.sleep(0.001)
                # boost fitting
                svm.adaboost(n_estimator, rec_frequency)
                svm.dumps(
                    open(os.path.join(self._pkl_path, '%sSVMClassifier.pkl' % k.title()), 'wb')
                )

            # eval model's capacity
            roc_auc_df = svm.roc_auc()
            acc_df = svm.accuracy()

            # record eval data
            self._dumps(roc_auc_df, name="SVMClassifier%sFlowROC_AUC")
            self._dumps(acc_df, name="SVMClassifier%sFlowAccuracy")

            if callback:
                # TODO: CALLBACK FUNCTION
                callback(self.reader)

            logger.info("CAVED!!!")


if __name__ == '__main__':
    e = EMBTASVMProcess()
    e.execute_classify_process(n_estimator=132, record_freq=10, training=0, best_n=-1)
