import os
import pickle

import numpy as np
import pandas as pd

from .const import package as c
from ..statUniversal import eq
from .StaUtills.Pfeatures import GenericFeaturesProcess
from .cal_imp_area import ReaderFromDimensions, logger

Reader = ReaderFromDimensions.init_from_pickle(c.DATA_DIR, c.PKL_FILE)

__pkl_path = Reader.pkl_path
__csv_path = Reader.csv_path
__DIMENSIONS = ['time', 'loc']

# __namespace = globals()
# __namespace[k.lower()] = val


def __load__(v):
    p = os.path.join(__pkl_path, v)
    with open(p, 'rb') as f:
        val = pickle.load(f)
    return val


def grouping(labels: np.ndarray):
    indices = np.arange(labels.shape[0])
    meta_keys = np.unique(labels[:, 0])
    sec_keys = np.unique(labels[:, 1])
    ret = []
    for meta_key in meta_keys:
        meta = np.argwhere(labels[:, 0] == meta_key).reshape(-1)
        for sec_key in sec_keys:
            sec = np.argwhere(labels[meta, 1] == sec_key).reshape(-1)
            ret.append(indices[meta][sec])
    return np.hstack(ret)


def select_signify(df):
    n = df.shape[0]
    t = n * c.THRESHOLD
    sigsco = df.copy()
    sigsco.loc[:, 'SIGNIFY'] = df.AREA > t

    sigdf = df[df.AREA > t]
    sid = sigdf.IDX.to_numpy(dtype=np.int32)
    sigds = Reader.dataset[:, sid]
    
    tl, ll = Reader.tlabels, Reader.llabels
    layer = np.array([s[0] for s in Reader.samples])
    merge = np.concatenate([tl.reshape(-1, 1), ll.reshape(-1, 1), layer.reshape(-1, 1), sigds], axis=1)

    grop_id = grouping(merge[:, 0:2])
    merge = merge[grop_id, :]
    title = np.hstack([np.array(['stage', 'layer', 'loci']), sid])
    expr = pd.DataFrame(merge, columns=title, index=None)
    print("SigScore: R[%s * %s] Expression: R[%s * %s]" % (*sigsco.shape, *expr.shape))
    return sigsco, expr


def transform_expr_and_sig_score():
    for k in __DIMENSIONS:
        logger.info("Working on Dimension@%s" % k.upper())
        df_score = __load__(c.IMPKL[k])
        score, expr = select_signify(df_score)

        p1 = os.path.join(__csv_path, 'Expr%sFlow.txt' % k.title())
        expr.to_csv(p1, sep='\t', header=True, index=False)

        p2 = os.path.join(__pkl_path, 'Expr%sFlow.pkl' % k.title())
        expr.to_pickle(p2)

        p3 = os.path.join(__csv_path, 'SigScore%sFlow.txt' % k.title())
        score.to_csv(p3, sep='\t', header=True, index=False)

        p4 = os.path.join(__pkl_path, 'SigScore%sFlow.pkl' % k.title())
        score.to_pickle(p4)


def accuracy_analysis(fitter, thd_df, dim):
    if isinstance(fitter, str):
        with open(fitter, 'rb') as f:
            fitter = GenericFeaturesProcess.load(f)

    # original features dataset
    xtr, xte, ytr, yte = fitter.train_or_test(dim)
    print("Original: R[%s * %s] || R[%s * %s]" % (*xtr.shape, *xte.shape))

    # significance features selected dataset
    sid = thd_df[thd_df.SIGNIFY == True].IDX.to_numpy(dtype=np.int32)
    sxtr, sxte = xtr[:, sid], xte[:, sid]
    print("Significant: R[%s * %s] || R[%s * %s]" % (*sxtr.shape, *sxte.shape))

    ret = list()

    def to_accdataframe(pre):
        acc_vec = eq(pre, yte, dim=0, int0=True)
        var_vec = np.arange(0, acc_vec.shape[0])
        acc_mtx = np.vstack([var_vec, var_vec, acc_vec]).T
        ret.append(pd.DataFrame(acc_mtx, columns=['Label', 'Predict', 'Value']))

    # predict from original test set
    pre_y = fitter.fit_.predict(xte)
    to_accdataframe(pre_y)

    # training new clf from eigen feature training set
    sclf = c.CLASSIFIER_MODEL(**c.RDF_PARAMS)
    sclf.fit(sxtr, ytr)
    # predict from eigen feature test set
    spre_y = sclf.predict(sxte)
    to_accdataframe(spre_y)

    return ret


def different_dim_accuracy_analysis():
    for k in __DIMENSIONS:
        logger.info("Working on Dimension@%s" % k.upper())

        fit_pkl, sig_pkl = c.FIT_PKL[k].split(':')
        fit = __load__(fit_pkl)
        threshold = __load__(sig_pkl)
        acc_df, sacc_df = accuracy_analysis(fit, threshold, k)

        p1 = os.path.join(__csv_path, 'Accuracy%sFlow.txt' % k.title())
        acc_df.to_csv(p1, sep='\t', header=True, index=False)

        p2 = os.path.join(__pkl_path, 'Accuracy%sFlow.pkl' % k.title())
        acc_df.to_pickle(p2)

        p3 = os.path.join(__csv_path, 'SAccuracy%sFlow.txt' % k.title())
        acc_df.to_csv(p3, sep='\t', header=True, index=False)

        p4 = os.path.join(__pkl_path, 'SAccuracy%sFlow.pkl' % k.title())
        acc_df.to_pickle(p4)


__METHODS = {
    "trans_and_sig": transform_expr_and_sig_score,
    "acc_between_select": different_dim_accuracy_analysis,
}


def execute_eim_analysis(*method):
    logger.info('Analysis with dataReader: %s' % Reader)
    for m in method:
        foo = __METHODS[m]
        logger.info("Lady's Processing %s" %
                    ' '.join([i.title() for i in foo.__name__.split('_')]))
        foo()
    logger.info("CAVED!!!")
