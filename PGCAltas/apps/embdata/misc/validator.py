import os

import numpy as np

import django
from django.conf import LazySettings

if os.environ.get("DJANGO_SETTINGS_MODULE", None) is None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "PGCAltas.settings.dev")
    django.setup()

settings = LazySettings()
PATH_ = os.path.dirname(settings.BASE_DIR) + settings.DATASET_URL

from PGCAltas.utils.StatExpr.Validator.uni import SFoldLayerCrossValidator
from embdata.misc.reader import BinomialDataReader
from ..data_const import DATA_DIR, VALIDATED_FILE





class EMBTABinomalValidator(SFoldLayerCrossValidator):

    s = 10
    data_reader_class = BinomialDataReader

    ROOT_PATH = PATH_
    PKL_DIR = "pickles"

    def __init__(self, dirname=None, pklfile=None):
        super().__init__(dirname or DATA_DIR, pklfile or VALIDATED_FILE)
        self.ctypes = None

    def get_reader(self):
        reader = self.get_reader_class().init_from_pickle(self.dirname, self.pklfile)
        return reader

    def resolute_from_reader(self):
        self.dataset, self.labels = self.reader.historic_trans['sigscreened']
        self.ctypes = self.dataset.loc[:, 'ctype'].to_numpy(dtype=np.str)

    def validate(self, **kwargs):
        raise_err = kwargs.pop('raise_err', True)
        self.wkfparams = kwargs
        files = self.pklfile
        for i in files:
            self.pklfile = i
            # config reader and params
            self._config_reader_and_params()
            # start validating
            self._validate(raise_err)
            break

    def validation_alg(self):
        labels = np.unique(self.ctypes)
        lidxs = [np.argwhere(self.ctypes == l).reshape(-1).astype(np.int) for l in labels]
        self.duplicate_loss(lidxs)
        queue = self.init_queue(lidxs)

        for _ in range(len(queue)):
            te_rows = np.array(queue.pop(0), dtype=np.int)
            tr_rows = np.hstack(queue).astype(np.int)
            yield tr_rows, te_rows
            queue.append(te_rows)
