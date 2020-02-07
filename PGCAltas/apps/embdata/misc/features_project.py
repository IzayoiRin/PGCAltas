import logging
import os
import numpy as np
import pandas as pd

from PGCAltas.utils.StatExpr.StatProcessor.FeaturesProcessor.processors import FeatureFilterExtractProcessor, \
    Feature2DViewerProcessor
from PGCAltas.utils.StatExpr.FunctionalDomain.EIMPAnalyzer import EIMAnalysis
from PGCAltas.utils.StatExpr.FunctionalDomain.EIMPCalculator import GenericEIMProcess
from PGCAltas.utils.StatExpr.FunctionalDomain.DimensionDenoiser import DimensionEstimate
from embdata.misc.reader import BinomialDataReader
from embdata.models import GenesInfo, CellsInfo
from ..data_const import POS, NEG, FILE_TYPE, \
    LDA_PARAMS, PCA_PARAMS, COMPONENTS_RATES, FILTER_RATES, TSNE_PARAMS, SPSVD_PARAMS


logger = logging.getLogger("django")


class EMBTABinomalEIMProcess(GenericEIMProcess):

    data_reader_class = BinomialDataReader
    test_size = 0.6

    def __init__(self, filename):
        super().__init__(filename, pos=POS + FILE_TYPE, neg=NEG + FILE_TYPE)

    def get_select_processor_class(self):
        self.screen_processor_class.spilter['test_size'] = self.test_size
        return self.screen_processor_class

    def pre_process(self):
        self.ppf = self.get_preprocessor()
        self.ppf(self.preprocesses, categories='auto')
        features = self.reader.features
        # record firstly passing data
        self.reader.dataset = self.ppf.dataset
        setattr(self.reader, 'encode_labels', self.ppf.labels)
        # push to historic stack as the recording of preprocessed transforming
        self.reader.historic_trans["preprocessed"] = (self.reader.dataset, self.reader.encode_labels)
        print("Recording PreProcessed data")
        self.reader.dumps_as_pickle(fname=self.pklfile)
        return features

    def importance_mtx(self, dimension, split_tt=False):
        features = self.pre_process()
        self.slp = self.get_screen_processor()
        self.slp(self.screen_process,
                 mparams=self.screen_process_params,
                 split=split_tt)
        mtx = self._imp_mtx_processing(features, dimension)
        return mtx


class EMBTABinomalEIMAnalysis(EIMAnalysis):

    data_reader_class = BinomialDataReader
    dimensions = ['binomial', ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def iter_genes(self):
        g_qs = GenesInfo.query.all().order_by('id')
        return iter(g_qs)

    def mark_sig_score(self, sigsco):
        g_qs = self.iter_genes
        gid2name = np.array([[q.id, q.name] for q in g_qs])

        sigsco.IDX = sigsco.IDX.astype(int)
        gid = sigsco.GENE.to_numpy(dtype=int)
        gname = gid2name[gid, 1]
        sigsco.loc[:, 'GENE'] = gname
        return sigsco

    def mark_sig_dataset(self, sigds, sid):
        a = self.reader.samples
        qs = CellsInfo.query.in_bulk(list(a))
        ctype = np.array([qs[int(cid)].type.name for cid in a])
        merge = np.concatenate([self.reader.labels.reshape(-1, 1), ctype.reshape(-1, 1), sigds], axis=1)
        title = np.hstack([['label', 'ctype'], sid])
        return merge, None, title

    def set_reader_data(self, data):
        print("Recording SigScreened data")
        self.reader.dataset = data[0]
        self.reader.historic_trans["sigscreened"] = (data[0], None)
        self.reader.dumps_as_pickle(fname=self.pklfile)


class EMBTABinomalDimensionEstimate(DimensionEstimate):

    data_reader_class = BinomialDataReader
    estimate_processor_class = FeatureFilterExtractProcessor
    estimate_process = ["PRINCIPAL_COMPONENTS", "LINEAR_DISCRIMINANT"]
    estimate_process_params = [PCA_PARAMS, LDA_PARAMS]
    test_size = 0.6
    dimension = ['binomial', ]

    viewer_class = Feature2DViewerProcessor
    viewer_method = ["T_STOCHASTIC", "SPARSE_SVD"]
    viewer_params = [TSNE_PARAMS, SPSVD_PARAMS]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_types = None
        self.neg_tag = None
        self.pos_tag = None

    def resolute_from_expr(self, expr):
        self.labels = expr.label.to_numpy(dtype=np.int8)
        self.sample_types = expr.ctype.to_numpy(dtype='U20')
        self.dataset = expr.iloc[:, 2: -1].copy().to_numpy(dtype=np.float32)
        whole_tag = expr.loc[:, ('label', 'ctype')].drop_duplicates()
        self.pos_tag = whole_tag.ctype\
            .to_numpy(dtype='U20')[whole_tag.label.astype(int) == self.kwargs.get('pos_lab', 1)]
        self.neg_tag = whole_tag.ctype\
            .to_numpy(dtype='U20')[whole_tag.label.astype(int) != self.kwargs.get('pos_lab', 1)]

        if self.n_componets is None:
            self.n_componets = np.math.floor(
                np.unique(self.sample_types).shape[0] * COMPONENTS_RATES
            )

        self.after_filter = self.kwargs.get('after_filter', np.math.floor(
            np.unique(self.labels).shape[0] * (1 - FILTER_RATES)
        ))

        if not self.kwargs.get('training', True):
            if hasattr(self.reader, 'tr_rows') and hasattr(self.reader, 'te_rows'):
                self.dataset = self.dataset[self.reader.tr_rows], self.dataset[self.reader.te_rows]
                self.labels = self.labels[self.reader.tr_rows], self.labels[self.reader.te_rows]

    def get_estimate_processor_class(self):
        self.estimate_processor_class.spilter['test_size'] = self.test_size
        return self.estimate_processor_class

    def estimate_dimension(self):
        # dataset splitted as train-test
        self.etp = self.get_estimate_processor()
        LDA_PARAMS['n_components'] = self.n_componets
        PCA_PARAMS['n_components'] = self.after_filter
        self.etp(*zip(self.estimate_process, self.estimate_process_params))

        # recombinate design matrix departed as training set and test set
        labels = self.etp.get_labels()
        tr = np.array([['tr', 1 if labels[i] in self.pos_tag else 0] for i in self.etp._trno])
        te = np.array([['te', 1 if labels[i] in self.pos_tag else 0] for i in self.etp._teno])
        self.etp.dataset = np.hstack([np.vstack([tr, te]), self.etp.dataset])
        header = ['set', 'label']
        header.extend(['D%s' % i for i in range(self.n_componets)])

        mtx = self._estimate_dimension(columns=header)
        print("%s Estimate: R[%s*%s] -----> R[%s*%s] supervised_acc=%.6f"
              % (self.kwargs['dim'].title(),
                 *self.dataset.shape,
                 mtx.shape[0], mtx.shape[1]-2,
                 self.etp.supervised_acc_)
              )
        return mtx

    def get_labels(self):
        return self.sample_types if self.kwargs.get('type') else self.labels

    def get_dataset(self):
        return self.dataset if self.kwargs.get('viewer') is None else self.estimated_dat_[self.kwargs['viewer']]

    def get_estimate_processor(self):
        self.kwargs['type'] = True
        p = super().get_estimate_processor()
        self.kwargs.pop('type')
        return p

    def viewer2d(self):
        rdataset = self.get_dataset()
        labels = rdataset.label.to_numpy(dtype=np.int8)
        title = rdataset.iloc[:, :2]
        dataset = rdataset.iloc[:, 2:]
        header = ['D0', 'D1']

        viewer = Feature2DViewerProcessor()
        TSNE_PARAMS['n_components'] = 2
        SPSVD_PARAMS['n_components'] = 2
        if self.kwargs.get('barnes_hut'):
            TSNE_PARAMS['method'] = 'barnes_hut'
            TSNE_PARAMS['angle'] = self.kwargs['barnes_hut']

        def _viewer2d(method, params):
            print("{viewer} 2D-Viewer: R[{m}*{n}] ----> R[{m}*2]"
                  .format(viewer=method, m=dataset.shape[0], n=dataset.shape[1]))
            # initial tsne-viewer design matrix
            viewer.init_from_data(dataset, labels)
            viewer(method, mparams=(params,))
            df = pd.DataFrame(viewer.dataset, index=rdataset.index, columns=header)
            df = pd.concat([title, df], axis=1)
            viewer.dumps(
                open(os.path.join(self._pkl_path, '%sEstimator.pkl' % method.title()), 'wb')
            )
            self._dumps(df, 'binomial', '%s{}_Estimated'.format(method))

        viewer_set = zip(self.viewer_method, self.viewer_params)
        for m, p in viewer_set:
            _viewer2d(m, p)
            if hasattr(viewer.fit_, 'kl_divergence_'):
                print("kl divergence: %s" % viewer.fit_.kl_divergence_)
            if hasattr(viewer.fit_, 'explained_variance_ratio_'):
                print("sum explained variance: %s" % viewer.fit_.explained_variance_ratio_.sum())

    def execute_estimate_process(self, viewer2d=True, **kwargs):
        super().execute_estimate_process(**kwargs)
        if self.kwargs.get('critical'):
            return
        if not viewer2d:
            return
        logger.info("Lady's Generating 2D-Viewer ...")
        self.kwargs['viewer'] = 0
        self.viewer2d()
        del self.kwargs['viewer']
