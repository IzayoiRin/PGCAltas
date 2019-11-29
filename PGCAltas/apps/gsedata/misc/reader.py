import re

import numpy as np
import pandas as pd

from PGCAltas.utils.StatExpr.DataReader.reader import DataReaderBase


class ReaderFromDimensions(DataReaderBase):

    def __init__(self, dirname, filename):
        super(ReaderFromDimensions, self).__init__(dirname, filename)
        self.samples = list()
        self.features = None
        self.labels_list_t = list()
        self.labels_list_l = list()
        self.llabels = None
        self.tlabels = None
        delattr(self, 'labels')
        delattr(self, 'labels_list')

    def workon(self, f_name, r_dframe):
        all_samples = r_dframe.columns
        pattern = re.compile(r"^\d(E?P)$")
        samples = [s for s in all_samples if re.match(pattern, s)]
        if not samples:
            return
        self.samples.extend(samples)
        locations = np.array([re.sub(pattern, lambda x: x.groups()[0], s) for s in samples])
        times = np.array([f_name.split('\\')[-1][1:4] for _ in range(len(samples))])
        df = r_dframe.loc[:, samples].T
        self.labels_list_l.append(locations)
        self.labels_list_t.append(times)
        if self.features is None:
            self.features = df.columns.to_numpy()
        return df.to_numpy()

    def get_ds_and_ls(self):
        self.dataset = np.concatenate(self.dataframes, axis=0)
        self.llabels = np.hstack(self.labels_list_l)
        self.tlabels = np.hstack(self.labels_list_t)
