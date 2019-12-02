import os
import pickle

import numpy as np
import pandas as pd

from PGCAltas.utils.StatExpr.DataReader.reader import DataReader, ReaderLoadError
from .const import package as c
from ..statUniversal import eq
from PGCAltas.utils.StatExpr.StaUtills.FeaturesProcessor.preprocessors import GenericFeaturesProcess
from .cal_imp_area import logger


class EIMAnalysis(object):

    data_reader_class = DataReader
    classifier = c.CLASSIFIER_MODEL
    classifier_params = c.RDF_PARAMS
    dimensions = None

    def __init__(self, dirname=None, pklfile=None, threshold=0.0,
                 pklfit=None, classifier=None, classifier_kwparams=None,
                 **rdparams):
        self.dirname = dirname or c.DATA_DIR
        self.pklfile = pklfile or c.PKL_FILE
        self.rdparams = rdparams
        self.reader = None
        self.__pkl_path, self.__csv_path = None, None

        self.threshold = threshold or c.THRESHOLD
        self.pklfit = pklfit or c.FIT_PKL

        if classifier:
            self.classifier = classifier
        if classifier_kwparams:
            self.classifier_params = classifier_kwparams

        self._METHODS = {
            "trans_and_sig": self.transform_expr_and_sig_score,
            "acc_between_select": self.different_dim_accuracy_analysis,
        }

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
        p = os.path.join(self.__pkl_path, v)
        with open(p, 'rb') as f:
            val = pickle.load(f)
        return val

    def transform_expr_and_sig_score(self):
        for k in self.dimensions:
            logger.info("Working on Dimension@%s" % k.upper())
            df_score = self.__load__(c.IMPKL[k])
            score, expr = self.select_signify(df_score)

            p1 = os.path.join(self.__csv_path, 'Expr%sFlow.txt' % k.title())
            expr.to_csv(p1, sep='\t', header=True, index=False)

            p2 = os.path.join(self.__pkl_path, 'Expr%sFlow.pkl' % k.title())
            expr.to_pickle(p2)

            p3 = os.path.join(self.__csv_path, 'SigScore%sFlow.txt' % k.title())
            score.to_csv(p3, sep='\t', header=True, index=False)

            p4 = os.path.join(self.__pkl_path, 'SigScore%sFlow.pkl' % k.title())
            score.to_pickle(p4)

    def select_signify(self, df):
        n = df.shape[0]
        t = n * self.threshold
        sigsco = df.copy()
        sigsco.loc[:, 'SIGNIFY'] = df.AREA > t
        sigsco = self.mark_sig_score(sigsco)

        sigdf = df[df.AREA > t]
        sid = sigdf.IDX.to_numpy(dtype=np.int32)
        sigds = self.reader.dataset[:, sid]

        merge, group_keys, title = self.mark_sig_dataset(sigds, sid)

        grop_id = self.grouping(group_keys, 0, 1)
        if grop_id is not None:
            merge = merge[grop_id, :]

        expr = pd.DataFrame(merge, columns=title, index=None)
        print("SigScore: R[%s * %s] Expression: R[%s * %s]" % (*sigsco.shape, *expr.shape))
        return sigsco, expr

    def mark_sig_score(self, sigsco):
        return sigsco

    def mark_sig_dataset(self, sigds, sid):
        group_keys = None
        return sigds, group_keys, sid

    @staticmethod
    def grouping(labels: np.ndarray, *order_keys, ordered=False, reverse=False):
        if not (order_keys and isinstance(labels, np.ndarray)):
            return

        indices = np.arange(labels.shape[0])

        f = lambda x: np.array(sorted(x, reverse=reverse)) if ordered else x
        keys = [f(np.unique(labels[:, idx])) for idx in order_keys]

        depth = len(order_keys) - 1
        ret = []

        def recurse_grouping(dp, up_idx, upper):
            for ck in keys[dp]:
                if up_idx is None:
                    cur_idx = np.argwhere(labels[:, order_keys[dp]] == ck).reshape(-1)
                else:
                    cur_idx = np.argwhere(labels[up_idx, order_keys[dp]] == ck).reshape(-1)
                if dp < depth:
                    recurse_grouping(dp+1, cur_idx, upper[cur_idx])
                else:
                    ret.append(upper[cur_idx])

        recurse_grouping(0, None, indices)

        return np.hstack(ret)

    def different_dim_accuracy_analysis(self):
        for k in self.dimensions:
            logger.info("Working on Dimension@%s" % k.upper())

            fit_pkl, sig_pkl = self.pklfit[k].split(':')
            fit = self.__load__(fit_pkl)
            threshold = self.__load__(sig_pkl)
            acc_df, sacc_df = self.accuracy_analysis(fit, threshold, k)

            p1 = os.path.join(self.__csv_path, 'Accuracy%sFlow.txt' % k.title())
            acc_df.to_csv(p1, sep='\t', header=True, index=False)

            p2 = os.path.join(self.__pkl_path, 'Accuracy%sFlow.pkl' % k.title())
            acc_df.to_pickle(p2)

            p3 = os.path.join(self.__csv_path, 'SAccuracy%sFlow.txt' % k.title())
            sacc_df.to_csv(p3, sep='\t', header=True, index=False)

            p4 = os.path.join(self.__pkl_path, 'SAccuracy%sFlow.pkl' % k.title())
            sacc_df.to_pickle(p4)

    def accuracy_analysis(self, fitter, thd_df, dim):
        if isinstance(fitter, str):
            with open(fitter, 'rb') as f:
                fitter = GenericFeaturesProcess.load(f)

        # original features dataset
        fitter.kwargs['dim'] = dim
        xtr, xte, ytr, yte = fitter.train_or_test()

        # significance features selected dataset
        sid = thd_df[thd_df.SIGNIFY == True].IDX.to_numpy(dtype=np.int32)
        sxtr, sxte = xtr[:, sid], xte[:, sid]

        ret = list()

        def to_accdataframe(pre):
            acc_vec = eq(pre, yte, dim=0, int0=True)
            var_vec = np.arange(0, acc_vec.shape[0])
            acc_mtx = np.vstack([var_vec, var_vec, acc_vec]).T
            print("Accuracy: %.6f" % np.mean(acc_vec))
            ret.append(pd.DataFrame(acc_mtx, columns=['Label', 'Predict', 'Value']))

        # predict from original test set
        print("Original: R[%s * %s] || R[%s * %s]" % (*xtr.shape, *xte.shape))
        pre_y = fitter.fit_.predict(xte)
        to_accdataframe(pre_y)

        # training new clf from eigen feature training set
        print("Significant: R[%s * %s] || R[%s * %s]" % (*sxtr.shape, *sxte.shape))
        sclf = self.classifier(**self.classifier_params)
        sclf.fit(sxtr, ytr)
        # predict from eigen feature test set
        spre_y = sclf.predict(sxte)
        to_accdataframe(spre_y)

        return ret

    def execute_eim_analysis(self, *method):
        self.reader = self.get_data_reader()
        if self.reader is None:
            raise ReaderLoadError("Can't load dataReader: %s" % self.data_reader_class.__name__)
        self.__pkl_path, self.__csv_path = self.get_file_path()

        logger.info('Analysis with dataReader: %s' % self.reader)

        for m in method:
            foo = self._METHODS[m]

            logger.info("Lady's Processing %s" %
                        ' '.join([i.title() for i in foo.__name__.split('_')]))

            foo()

        logger.info("CAVED!!!")
