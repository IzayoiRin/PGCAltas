import os
import io
import re
import numpy as np
import pandas as pd
import pickle

from PGCAltas.utils.webUniversal import base64UUID
from PGCAltas.utils.errors import OSPathError, DataFrameFitError, ReaderLoadError
from .. import PATH_


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

    NAME = 'OBJ-%s-%s.pkl'

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
        # historic transformed hash stack
        self.historic_trans = dict()

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

    def dumps_as_pickle(self, fname=None, **pklkwargs):
        """
        name:
            base64 ---> uuid (inner)
            None ---> uuid (inner), base64 (outter)
        """
        if not pklkwargs:
            if fname is None:
                from datetime import datetime
                t = datetime.now().strftime('%m%d')
                uuid, b64 = base64UUID()()
                name = self.NAME % (t, uuid)  # uuid
                outtername = "%s-%s" % (t, b64)
            else:
                outtername = fname
                # base64 ----> uuid
                t, b64 = fname.split('-', maxsplit=2)
                name = self.NAME % (t, base64UUID('de')(b64))  # uuid

            p = os.path.join(self.pkl_path, name)
            with open(p, 'wb') as f:
                pickle.dump(self, f, -1)

            # record denovo pklfile's name
            setattr(self, 'pklname', outtername)
            print('Pickled: Done')
            return

        # for attrname, pklname in pklkwargs.items():
        #     name = '%s%s-%s.pkl' % (pklname, t, uuid)
        #     p = os.path.join(self.pkl_path, name)
        #     with open(p, 'wb') as f:
        #         pickle.dump(getattr(self, attrname), f, -1)
        # print('Done')

    def loads_from_pickle(self, **pklkwargs):
        for attrname, pklname in pklkwargs.items():
            p = os.path.join(self.pkl_path, pklname)
            if not os.path.exists(p):
                raise OSPathError('Could not find {name} in {path}'.format(name=pklname, path=self.path))
            with open(p, 'rb') as f:
                setattr(self, attrname, pickle.load(f))

    @classmethod
    def init_from_pickle(cls, dirname, pklfile):
        t, b64 = pklfile.split('-', maxsplit=2)
        pklfile = cls.NAME % (t, base64UUID('de')(b64))
        path = os.path.join(cls.ROOT_PATH, dirname, cls.PKL_DIR, pklfile)
        if not os.path.exists(path):
            raise OSPathError('Could not find {name} in {path}'.format(name=pklfile, path=path))
        print('LOAD FROM: %s' % path)
        with open(path, 'rb') as f:
            self = pickle.load(f)
        setattr(self, 'pklname_', pklfile)
        return self

    def get_ds_and_ls(self):
        raise NotImplementedError

    def __str__(self):
        return "dataReader@{name}_Non_STData".format(name=self.path.split('/')[-1]) \
            if isinstance(self.dataset, list) \
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
        # push to historic stack as the recording of original transforming
        self.historic_trans['original'] = (self.dataset, self.labels)


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
