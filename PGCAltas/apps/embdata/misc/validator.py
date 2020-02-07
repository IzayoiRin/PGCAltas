import numpy as np

from PGCAltas.utils.StatExpr.DataReader.reader import ReaderLoadError
from embdata.misc.reader import BinomialDataReader


class GenericValidator(object):

    data_reader_class = BinomialDataReader
    model = None

    def __init__(self, dirname=None, pklfile=None):
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
        pkl_path, csv_path = self.reader.pkl_path, self.reader.csv_path
        return pkl_path, csv_path

    def resolute_from_reader(self):
        # dataset, labels M*N
        self.dataset, self.labels = self.reader.dataset, self.reader.labels

    def validate(self):
        self.reader = self.get_reader()
        if self.reader is None:
            raise ReaderLoadError("Can't load dataReader: %s" % self.data_reader_class.__name__)
        self._pkl_path, self._csv_path = self.get_file_path()
        self.resolute_from_reader()
        if not hasattr(self, 'validation_alg'):
            return
        diter = self.validation_alg()
        import copy
        reader = copy.copy(self.reader)
        for xy in diter:
            reader.tr_rows, reader.te_rows = xy
            print(reader.dataset[reader.tr_rows, :].shape)


class SFoldCrossMixin(object):

    placeholder = None

    def init_queue(self):
        # dataset M*N
        m, n = self.dataset.shape
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
            yield te_rows, tr_rows
            queue.append(te_rows)


class SFoldCrossValidator(GenericValidator, SFoldCrossMixin):

    s = 10


if __name__ == '__main__':
    DATA_DIR = "EMTAB6967"
    PKL_FILE = 'OBJEMTAB696719121616_small.pkl'
    sfc_validator = SFoldCrossValidator(DATA_DIR, PKL_FILE)
    sfc_validator.validate()
