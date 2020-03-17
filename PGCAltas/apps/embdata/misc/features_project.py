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
    LDA_PARAMS, PCA_PARAMS, COMPONENTS_RATES, FILTER_RATES, TSNE_PARAMS, SPSVD_PARAMS, ENSEMBLE_LDA_CLASSIFIER_PKL


logger = logging.getLogger("django")


class EMBTABinomalEIMProcess(GenericEIMProcess):

    data_reader_class = BinomialDataReader
    test_size = 0.6

    def __init__(self, filename, **kwargs):
        super().__init__(filename, pos=POS + FILE_TYPE, neg=NEG + FILE_TYPE, **kwargs)

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
        print("Initiating Random Forest Selector")
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
        self.reader.historic_trans["sigscreened"] = (data[0], self.reader.labels)
        self.reader.dumps_as_pickle(fname=self.pklfile)


class EMBTABinomalDimensionEstimate(DimensionEstimate):

    data_reader_class = BinomialDataReader
    estimate_processor_class = FeatureFilterExtractProcessor
    estimate_process = ["PRINCIPAL_COMPONENTS", "LINEAR_DISCRIMINANT"]
    estimate_process_params = [PCA_PARAMS, LDA_PARAMS]
    test_size = 0.2
    dimension = ['binomial', ]

    viewer_class = Feature2DViewerProcessor
    viewer_method = ["T_STOCHASTIC", "SPARSE_SVD"]
    viewer_params = [TSNE_PARAMS, SPSVD_PARAMS]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_types = None
        self.neg_tag = None
        self.pos_tag = None
        self.primary_dim = None

    def get_data_reader(self):
        if self.reader is None:
            return super().get_data_reader()
        # overwritten for validator: return reader with 'sigscreened' version and splitted row_no
        self.pklfile = self.reader.pklname
        return self.reader

    def resolute_from_expr(self, expr):
        """
        training -- 1(training model) or -1(validating model) & tr/te rows -- 1(splitted):
            self.dataset, self.labels, self.sample_types : (tr, te)

        training -- 1(training model) & tr/te rows -- 0(not splitted) or training -- 0(predicting model):
            self.dataset, self.labels, self.sample_types : arrays
        :param expr: historic_trans['sigscreened']
        """
        self.labels = expr.label.to_numpy(dtype=np.int8)
        self.sample_types = expr.ctype.to_numpy(dtype='U20')
        self.dataset = expr.iloc[:, 2: -1].copy().to_numpy(dtype=np.float32)  # type: np.ndarray
        self.primary_dim = self.dataset.shape
        sample_counts = self.primary_dim[0]
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

        if self.kwargs['training'] in [1, -1]:
            # New training without no split
            if self.kwargs['training'] == 1 and hasattr(self.reader, 'te_rows'):
                delattr(self.reader, 'tr_rows')
                delattr(self.reader, 'te_rows')
            # Training or Validating with split
            if hasattr(self.reader, 'tr_rows') and hasattr(self.reader, 'te_rows'):
                self.dataset = self.dataset[self.reader.tr_rows], self.dataset[self.reader.te_rows]
                self.labels = self.labels[self.reader.tr_rows], self.labels[self.reader.te_rows]
                self.sample_types = self.sample_types[self.reader.tr_rows], self.sample_types[self.reader.te_rows]

                sample_counts = min(len(self.reader.tr_rows), len(self.reader.te_rows))

        self.after_filter = min(self.after_filter, self.primary_dim[1], sample_counts)

    def get_estimate_processor_class(self):
        self.estimate_processor_class.spilter['test_size'] = self.test_size
        return self.estimate_processor_class

    def estimate_dimension(self):
        # dataset splitted as train-test
        self.etp = self.get_estimate_processor()
        # config params
        LDA_PARAMS['n_components'] = self.n_componets
        PCA_PARAMS['n_components'] = self.after_filter

        # training model(1) and validating model(-1)
        if self.kwargs['training']:
            # start estimate process
            etp_fit = self.etp(*zip(self.estimate_process, self.estimate_process_params))

            # recombinate design matrix departed as training set and test set
            labels = self.etp.get_labels()
            tr = np.array([['tr', 1 if labels[i] in self.pos_tag else 0] for i in self.etp._trno])
            te = np.array([['te', 1 if labels[i] in self.pos_tag else 0] for i in self.etp._teno])
            cur_dim = self.etp.dataset.shape
            self.etp.dataset = np.hstack([np.vstack([tr, te]), self.etp.dataset])
            # reader record te / tr rows
            self.reader.tr_rows = self.etp._trno
            self.reader.te_rows = self.etp._teno
            # record fitted LDA model in training model
            self._pickled(etp_fit)
        else:
            # load fitted LDA model from pkl
            etp_fit = self.__load__(ENSEMBLE_LDA_CLASSIFIER_PKL[self.kwargs['dim']])

            # start estimate process
            self.etp(*zip(self.estimate_process, self.estimate_process_params), training=False, loaded_fit=etp_fit)

            # recombinate design matrix departed as test set
            labels = self.etp.get_labels()
            cur_dim = self.etp.dataset.shape
            te = np.array([['te', 1 if i in self.pos_tag else 0] for i in labels])
            self.etp.dataset = np.hstack([te, self.etp.dataset])

        # set header of estimated data frame
        header = ['set', 'label']
        header.extend(['D%s' % i for i in range(cur_dim[1])])

        # built as DataFrame
        mtx = self._estimate_dimension(columns=header)
        print("%s Estimate: R[%s*%s] -----> R[%s*%s] supervised_acc=%.6f"
              % (self.kwargs['dim'].title(),
                 *self.primary_dim,
                 *cur_dim,
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
            self._dumps(df, '%s{}_Estimated'.format(method))

        viewer_set = zip(self.viewer_method, self.viewer_params)
        for m, p in viewer_set:
            _viewer2d(m, p)
            if hasattr(viewer.fit_, 'kl_divergence_'):
                print("kl divergence: %s" % viewer.fit_.kl_divergence_)
            if hasattr(viewer.fit_, 'explained_variance_ratio_'):
                print("sum explained variance: %s" % viewer.fit_.explained_variance_ratio_.sum())

    def execute_estimate_process(self, viewer2d=True, callback=None, **kwargs):
        super().execute_estimate_process(callback=callback, **kwargs)
        if self.kwargs.get('critical'):
            return
        if not viewer2d:
            return
        logger.info("Lady's Generating 2D-Viewer ...")
        self.kwargs['viewer'] = 0
        self.viewer2d()
        del self.kwargs['viewer']
