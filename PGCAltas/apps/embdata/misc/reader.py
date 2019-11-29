import os
import re
import numpy as np
import pandas as pd


from PGCAltas.utils.StatExpr.DataReader.reader import DataReader


class BinomialDataReader(DataReader):

    def __init__(self, dirname, filename, **lab_pattern):
        super().__init__(dirname, filename)
        self.neg_p = re.compile(lab_pattern.get('neg'))
        self.pos_p = re.compile(lab_pattern.get('pos'))
        self.samples = list()
        self.samples_list = list()

    def workon(self, f_name, r_dframe):
        """
        resolute data set and label set from flag pattern
        :param f_name: file_path
        :param r_dframe: N * m
        :return:
        """
        m = r_dframe.shape[1]
        file_flag = os.path.split(f_name)[-1]

        for idx, i in enumerate(['neg_p', 'pos_p']):
            if re.match(getattr(self, i), file_flag):
                labels = np.array([idx or -1 for _ in range(m)])
                break
        else:
            return

        if self.features is None:
            self.features = r_dframe.index.to_numpy()
        self.samples_list.append(r_dframe.columns.to_numpy())
        self.labels_list.append(labels)
        return r_dframe.to_numpy(dtype=np.int32).T

    def get_ds_and_ls(self):
        super().get_ds_and_ls()
        self.samples = np.hstack(self.samples_list)
