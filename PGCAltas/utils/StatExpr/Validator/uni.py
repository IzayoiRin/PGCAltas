import os
import logging

import numpy as np

from PGCAltas.utils.StatExpr.DataReader.reader import DataReader
from PGCAltas.utils.errors import ReaderLoadError, WorkFlowInitiationError, WorkFlowRunningError

logger = logging.getLogger("django")


class GenericValidator(object):
    data_reader_class = DataReader
    s = 1
    workflow = None

    def __init__(self, dirname, pklfile):
        self.dirname = dirname
        self.pklfile = pklfile

        self.reader = None
        self._pkl_path, self._csv_path = None, None

        self.dataset = None
        self.labels = None

    def get_reader_class(self):
        return self.data_reader_class

    def get_reader(self):
        reader = self.get_reader_class().init_from_pickle(self.dirname, self.pklfile)
        return reader

    def get_file_path(self):
        vdir = "_".join(['validated', self.reader.pklname_.rsplit('.', 1)[0]])
        path = os.path.join(self.reader.path, vdir)
        pkl_path = os.path.join(path, 'pickles')
        csv_path = os.path.join(path, 'texts')
        return pkl_path, csv_path

    def resolute_from_reader(self):
        # dataset, labels M*N
        self.dataset, self.labels = self.reader.historic_trans['sigscreened']

    def start_workflow(self, reader):
        if self.workflow is None:
            raise WorkFlowInitiationError
        workflow = self.workflow(reader)
        workflow(**self.wkfparams)

    def validation_alg(self):
        raise NotImplementedError

    def validate(self, **kwargs):
        raise_err = kwargs.pop('raise_err', True)
        self.wkfparams = kwargs

        # config reader and params
        self._config_reader_and_params()
        # start validating
        self._validate(raise_err)

    def _config_reader_and_params(self):
        # load reader
        self.reader = self.get_reader()
        if self.reader is None:
            raise ReaderLoadError("Can't load dataReader: %s" % self.data_reader_class.__name__)
        logger.info('%s Complete Ready' % self.reader)

        # resolute file path
        self._pkl_path, self._csv_path = self.get_file_path()
        # resolute dataset and label from reader
        self.resolute_from_reader()

    def _validate(self, raise_err):

        logger.info("Lady's Initializing Queue ...")
        # core alg of validator
        diter = self.validation_alg()

        s = 1

        logger.info("Lady's Validating Data ...")

        for xy in diter:

            import copy
            reader = copy.copy(self.reader)
            reader.dataset, reader.labels = reader.historic_trans['sigscreened']

            reader.tr_rows, reader.te_rows = xy
            tr, te, tol = len(xy[0]), len(xy[1]), len(xy[0]) + len(xy[1])
            print("Validating Folds: %d / %d \ntr / te [tol] : %d / %d [%d]" % (s, self.s, tr, te, tol))

            v = 'v%d' % s
            reader.pkl_path = os.path.join(self._pkl_path, v)
            if not os.path.exists(reader.pkl_path):
                os.makedirs(reader.pkl_path)

            reader.csv_path = os.path.join(self._csv_path, v)
            if not os.path.exists(reader.csv_path):
                os.makedirs(reader.csv_path)

            try:
                self.start_workflow(reader)
            except Exception as e:
                logging.critical(e)
                if raise_err:
                    raise WorkFlowRunningError(e)
                break

            s += 1


class SFoldCrossMixin(object):

    placeholder = None

    def init_queue(self):
        # dataset M*N
        m = self.labels.shape[0]
        b, a = divmod(m, self.s)
        # get row index
        idx = list(range(m))
        np.random.shuffle(idx)
        if a:
            idx.extend([self.placeholder for _ in range(self.s - a)])
        queue = np.array(idx).reshape(self.s, -1).tolist()
        while self.placeholder in queue[-1]:
            queue[-1].remove(self.placeholder)
        return queue

    def validation_alg(self):
        queue = self.init_queue()
        for _ in range(len(queue)):
            te_rows = queue.pop(0)
            tr_rows = np.hstack(queue)
            yield tr_rows, te_rows
            queue.append(te_rows)


class SFoldLayerCrossMixin(object):

    placeholder = np.nan
    search_placeholder = np.isnan

    def init_queue(self, lidxs):
        layers_queue = [self._init_queue(sidxs) for sidxs in lidxs]
        queue = [list() for _ in range(self.s)]
        for lq in layers_queue:
            for i, lidx in enumerate(lq):
                queue[i].extend(lidx)
        return queue

    def _init_queue(self, sidxs):
        m = sidxs.shape[0]
        b, a = divmod(m, self.s)
        # get row index
        np.random.shuffle(sidxs)
        if a:
            sidxs = np.hstack([sidxs, [self.placeholder for _ in range(self.s - a)]])
        queue = sidxs.reshape(-1, self.s).T.tolist()
        phlocs = np.argwhere(self.search_placeholder(queue))
        for row, col in phlocs:
            queue[row].pop(col)
        return queue

    def duplicate_loss(self, lidxs):
        nk = np.array([idxs.shape[0] for idxs in lidxs], dtype=np.int)
        dups = np.argwhere(nk < self.s).reshape(-1)
        for l in dups:
            idxs = lidxs[l]
            duped_idxs = np.random.choice(idxs, size=self.s)
            lidxs[l] = duped_idxs

    def validation_alg(self):
        labels = np.unique(self.labels)
        lidxs = [np.argwhere(self.labels == l).reshape(-1).astype(np.int) for l in labels]

        self.duplicate_loss(lidxs)

        queue = self.init_queue(lidxs)

        for _ in range(len(queue)):
            te_rows = queue.pop(0)
            tr_rows = np.hstack(queue)
            yield tr_rows, te_rows
            queue.append(te_rows)


class SFoldCrossValidator(SFoldCrossMixin, GenericValidator):
    """
    from embdata.data_const import PKL_FILE, DATA_DIR
    sfc_validator = SFoldCrossValidator(DATA_DIR, PKL_FILE)
    sfc_validator.workflow = ValidationWorkFlow
    sfc_validator.validate()
    """
    s = 10


class SFoldLayerCrossValidator(SFoldLayerCrossMixin, GenericValidator):
    """
    from embdata.data_const import PKL_FILE, DATA_DIR
    sfc_validator = SFoldCrossValidator(DATA_DIR, PKL_FILE)
    sfc_validator.workflow = ValidationWorkFlow
    sfc_validator.validate()
    """
    s = 10
