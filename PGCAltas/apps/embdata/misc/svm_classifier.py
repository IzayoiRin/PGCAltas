import os
import pickle
import time

from tqdm import tqdm

import numpy as np
import pandas as pd

from PGCAltas.utils.StatExpr.DataReader.reader import ReaderLoadError
from PGCAltas.utils.StatExpr.StatProcessor.ClassiferProcessor.trad_classifier import SupportVectorMachineClassifier
from embdata.data_const import DATA_DIR, PKL_FILE, ESTIMATED_EXPR_PKL, SVM_PARAMS
from embdata.misc.reader import BinomialDataReader


class EMBTASVMClassifier(SupportVectorMachineClassifier):

    def __init__(self):
        super().__init__()

        self.roc_mtxs = list()
        self.auc_arr = list()
        self.acc_arr = list()
        self.acc_mtx = list()

        self.records = None

    def get_model(self):
        return self.model_class(**SVM_PARAMS)

    def acc_bar(self, y):
        acc = np.equal(self.ypre_, y).astype(np.int8)
        return acc, np.mean(acc)

    def __boost(self, n):
        self.n_boost = n
        self.fit_model()
        self.predict()
        acc_m, acc = self.acc_bar(self.yte)
        self.acc_arr.append(acc)
        self.acc_mtx.append(acc_m)

        roc, auc = self.roc(), self.auc()

        self.roc_mtxs.append(roc)
        self.auc_arr.append(auc)

        return acc, auc

    def adaboost(self, n_estimator: int, record_freq=None):
        if record_freq is None:
            self.records = [n_estimator]
            self.__boost(n_estimator)
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
        # ret = [ 4*N, ...]
        sta_mtx = np.hstack(ret).T
        df = pd.DataFrame(sta_mtx, columns=["Boost", "AUC", "FPR", "TPR"])
        return df

    def accuracy(self):
        ret = list()
        for idx, n in enumerate(self.records):
            acc = self.acc_arr[idx]
            acc_m = self.acc_mtx[idx]
            temp = [
                np.array([n for _ in acc_m]),
                np.array([acc for _ in acc_m]),
                np.array([i + 1 for i in range(len(acc_m))]),
                acc_m,
            ]
            ret.append(np.vstack(temp))
        # ret = [ 4*N, ...]
        sta_mtx = np.hstack(ret).T
        df = pd.DataFrame(sta_mtx, columns=["Boost", "Acc", "Case", "Predict"])
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
        self.dataset = expr.to_numpy()
        self.labels = expr.index.to_numpy(dtype=np.int8)

    def get_classifier_class(self):
        self.classifier_class.test_size = self.test_size
        return self.classifier_class

    def get_classifier(self):
        cls = self.get_classifier_class()
        obj = cls()
        obj.initial_from_data(self.get_dataset(), self.get_labels())
        # from sklearn.datasets import make_hastie_10_2
        # x, y = make_hastie_10_2(n_samples=1007, random_state=0)
        # obj.initial_from_data(x, y)
        return obj

    def get_labels(self):
        return self.labels

    def get_dataset(self):
        return self.dataset

    def _dumps(self, datframe, dim, name):
        p1 = os.path.join(self._csv_path, name % dim.title() + '.txt')
        datframe.to_csv(p1, sep='\t', header=True, index=True)

        p2 = os.path.join(self._pkl_path, name % dim.title() + '.pkl')
        datframe.to_pickle(p2)

    def execute_classify_process(self, **kwargs):
        self.reader = self.get_data_reader()
        if self.reader is None:
            raise ReaderLoadError("Can't load dataReader: %s" % self.data_reader_class.__name__)
        self._pkl_path, self._csv_path = self.get_file_path()

        self.kwargs = kwargs
        n_estimator = kwargs.pop("n_estimator", 10)
        rec_frequency = kwargs.pop("record_freq", None)
        for k in self.dimension:
            self.kwargs['dim'] = k
            expr = self.__load__(ESTIMATED_EXPR_PKL[k])
            self.resolute_from_expr(expr)
            # init svm classifier
            svm = self.get_classifier()
            print(svm.xtr.shape, svm.xte.shape)
            time.sleep(0.001)
            # boost fitting
            svm.adaboost(n_estimator, rec_frequency)
            svm.dumps(
                open(os.path.join(self._pkl_path, '%sSVMClassifier.pkl' % k.title()), 'wb')
            )
            # eval model's capacity
            roc_auc_df = svm.roc_auc()
            acc_df = svm.accuracy()
            print(roc_auc_df)
            print(acc_df)
            # record eval data
            self._dumps(roc_auc_df, k, name="SVMClassifier%sFlowROC_AUC")
            self._dumps(acc_df, k, name="SVMClassifier%sFlowAccuracy")


if __name__ == '__main__':
    e = EMBTASVMProcess()
    e.execute_classify_process(n_estimator=132, record_freq=10)
