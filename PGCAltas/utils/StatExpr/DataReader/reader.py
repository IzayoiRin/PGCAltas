import os
import io
import re
import numpy as np
import pandas as pd
import pickle

from .. import PATH_


class OSPathError(Exception):
    pass


class DataFrameFitError(Exception):
    pass


class ReaderLoadError(Exception):
    pass


class DataReaderBase(object):
    """
    Examples:
        r = DataReader('GSE120963', r'.*?\.txt$')
        r.read(header=0, sep='\t', index_col=0).get_ds_and_ls()
        print(r.dataset.shape)
        print(r.labels.shape)

        r.dumps_as_pickle(dataset='DS19103016.pkl', labels='LS19103016.pkl')
        r.loads_from_pickle(dataset='DS19103016.pkl', labels='LS19103016.pkl')
        print(r.dataset.shape)
        print(r.labels.shape)

        r.dumps_as_pickle()
        r = ReadData.init_from_pickle('GSE120963', "OBJGSE12096319103016.pkl")
        print(r.dataset.shape)
        print(r.labels.shape)

    Overwrite methods:
        self.workon()
        self.get_ds_and_ls()
    """

    ROOT_PATH = PATH_
    PKL_DIR = "pickles"
    CSV_DIR = 'texts'

    def __init__(self, dirname, filename):
        path = os.path.join(self.ROOT_PATH, dirname)
        if not os.path.exists(path):
            raise OSPathError('Could not find {name} in {path}'.format(name=dirname, path=path))

        self.path = path
        self.pkl_path = os.path.join(self.path, self.PKL_DIR)
        self.csv_path = os.path.join(self.path, self.CSV_DIR)

        file_pattern = re.compile(filename)
        file_names = [f for f in os.listdir(self.path) if re.match(file_pattern, f)]
        self.files = [os.path.join(self.path, f) for f in file_names]

        self.dataframes = list()
        self.labels_list = list()
        self.dataset = list()
        self.labels = list()

        self.__flushed = False

    def read(self, **kwargs):
        for f in self.files:
            dataframe = self._read(f, **kwargs)
            if dataframe is None:
                continue
            print(f)
            self.dataframes.append(dataframe)
        self.__flushed = True
        return self

    def _read(self, f, **kwargs):
        rdf = pd.read_table(f, **kwargs)
        try:
            df = self.workon(f, rdf)
        except Exception as e:
            raise DataFrameFitError(e)
        return df

    def workon(self, f_name, r_dframe):
        return r_dframe

    def info(self):
        if self.__flushed:
            self._info_rets = list()
            buffer = io.StringIO()
            for df in self.dataframes:
                df.info(buf=buffer)
                self._info_rets.append(buffer.getvalue())
        for r in self._info_rets:
            print(r)

    def dumps_as_pickle(self, **pklkwargs):
        from datetime import datetime
        t = datetime.now().strftime('%y%m%d%H')

        if not pklkwargs:
            p = os.path.join(self.pkl_path,
                             'OBJ%s%s.pkl' % (self.path.split('/')[-1], t))
            with open(p, 'wb') as f:
                pickle.dump(self, f, -1)
            print('Done')
            return

        for attrname, pklname in pklkwargs.items():
            p = os.path.join(self.pkl_path, '%s%s.pkl' % (pklname, t))
            with open(p, 'wb') as f:
                pickle.dump(getattr(self, attrname), f, -1)
        print('Done')

    def loads_from_pickle(self, **pklkwargs):
        for attrname, pklname in pklkwargs.items():
            p = os.path.join(self.pkl_path, pklname)
            if not os.path.exists(p):
                raise OSPathError('Could not find {name} in {path}'.format(name=pklname, path=self.path))
            with open(p, 'rb') as f:
                setattr(self, attrname, pickle.load(f))

    @classmethod
    def init_from_pickle(cls, dirname, pklfile):
        path = os.path.join(cls.ROOT_PATH, dirname, cls.PKL_DIR, pklfile)
        if not os.path.exists(path):
            raise OSPathError('Could not find {name} in {path}'.format(name=dirname, path=path))
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_ds_and_ls(self):
        raise NotImplementedError

    def __str__(self):
        return "dataReader@{name}_Non_data" \
            if self.dataset is [] \
            else "dataReader@{name}_R[{row} * {col}]".format(name=self.path.split('/')[-1],
                                                             row=self.dataset.shape[0],
                                                             col=self.dataset.shape[1])


class DataReader(DataReaderBase):
    """
    Examples:
        r = DataReader('GSE120963', r'.*?\.txt$')
        r.read(header=0, sep='\t', index_col=0).get_ds_and_ls()
        print(r.dataset.shape)
        print(r.labels.shape)

        r.dumps_as_pickle(dataset='DS', labels='LS')
        r.loads_from_pickle(dataset='DS19103017.pkl', labels='LS19103017.pkl')
        print(r.dataset.shape)
        print(r.labels.shape)

        r.dumps_as_pickle()
        r = DataReader.init_from_pickle('GSE120963', "OBJGSE12096319103017.pkl")
        print(r.dataset.shape)
        print(r.labels.shape)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = None

    def workon(self, f_name, r_dframe):
        """
        resolute dataset and labelset from file_name and row_dataframe
        :param f_name: file_path
        :param r_dframe: N * m
        :return: Design Matrix m * N
        """
        if self.features is None:
            self.features = r_dframe.index.to_numpy()
        self.labels_list.append(r_dframe.columns.to_numpy())
        return r_dframe.to_numpy().T

    def get_ds_and_ls(self):
        self.dataset = np.concatenate(self.dataframes, axis=0)
        self.labels = np.hstack(self.labels_list)


def _example():
    """
    An User's guidance for the DataReader
    """
    r = DataReader('GSE120963', r'.*?\.txt$')
    r.read(header=0, sep='\t', index_col=0).get_ds_and_ls()
    print(r.dataset.shape)
    print(r.labels.shape)

    r.dumps_as_pickle(dataset='DS', labels='LS')
    r.loads_from_pickle(dataset='DS19103017.pkl', labels='LS19103017.pkl')
    print(r.dataset.shape)
    print(r.labels.shape)

    r.dumps_as_pickle()
    r = DataReader.init_from_pickle('GSE120963', "OBJGSE12096319103017.pkl")
    print(r.dataset.shape)
    print(r.labels.shape)
