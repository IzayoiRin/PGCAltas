import os
import logging
import pickle

from torch.utils.data import Dataset

from pgc_torch.errors import OSPathError
from . import PATH_, TORCH_PATH_

logger = logging.getLogger('django')


class EMTABaseDataset(object):

    LROOT_PATH = PATH_
    LROOT_DIR = 'EMTAB6967'
    ROOT_PATH = TORCH_PATH_

    ROOT_DIR = 'data'
    PKL_DIR = "pickles"
    CSV_DIR = 'texts'

    def __init__(self, raw_pklfile, **resolute):
        raw_path = os.path.join(self.LROOT_PATH, self.LROOT_DIR)
        if not os.path.exists(raw_path):
            raise OSPathError('Could not find {name} in {path}'.format(name=self.LROOT_DIR, path=raw_path))
        self.raw_path = raw_path
        self.raw_pkl_path = os.path.join(self.raw_path, self.PKL_DIR)
        self.rdata = self._load_raw_data_from_pickle(raw_pklfile)

        self.path = os.path.join(self.ROOT_PATH, self.ROOT_DIR)
        self.pkl_path = os.path.join(self.path, self.PKL_DIR)
        self.csv_path = os.path.join(self.path, self.CSV_DIR)
        self._init_dir(self.pkl_path, self.csv_path)

        self.resolute_dat_(**resolute)

    @staticmethod
    def _init_dir(*path):
        for p in path:
            if not os.path.exists(p):
                os.makedirs(p)

    def _load_raw_data_from_pickle(self, pklfile):
        path = os.path.join(self.raw_pkl_path, pklfile)
        if not os.path.exists(path):
            raise OSPathError('Could not find {name} in {path}'.format(name=pklfile, path=path))
        logger.debug('LOAD FROM: %s' % path)
        with open(path, 'rb') as f:
            return pickle.load(f)

    def resolute_dat_(self, **kwargs):
        raise NotImplementedError

    def dumps_as_pickle(self, fname):
        p = os.path.join(self.pkl_path, fname)
        with open(p, 'wb') as f:
            pickle.dump(self, f, -1)
        setattr(self, 'pklname', fname)
        print('Pickled: Done')

    @classmethod
    def init_from_pickle(cls, pklfile):
        path = os.path.join(cls.ROOT_PATH, cls.ROOT_DIR, cls.PKL_DIR, pklfile)
        if not os.path.exists(path):
            raise OSPathError('Could not find {name} in {path}'.format(name=pklfile, path=path))
        logger.debug('LOAD FROM: %s' % path)
        with open(path, 'rb') as f:
            self = pickle.load(f)
        setattr(self, 'pklname_', pklfile)
        return self
