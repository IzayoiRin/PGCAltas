import numpy as np
import pandas as pd

from embdata.misc import *
from embdata.misc.EchartSunburst import NodeArray


class StaticCellCounts(object):

    def __init__(self, file):
        self.file = os.path.join(ROOT_, file)
        self.matrix = None
        self.datframe = None

    def _initialize(self, *shape):
        if self.matrix is None:
            self.matrix = np.zeros(shape, dtype=np.int32)

    def build_dataframe(self, header=True, seq='\t'):
        assert os.path.exists(self.file)
        text_matrix = list()
        with open(self.file, 'r') as f:
            line = f.readline() if header else 1
            while True:
                line = f.readline()
                ret = line.strip().split(seq)
                if not all(ret):
                    break
                text_matrix.append([ret[0], ret[2], ret[4]])

        text_matrix = np.array(text_matrix, dtype=np.int32) - np.array([1, 1, 0])
        stages = np.unique(text_matrix[:, 0]) + 1
        types = np.unique(text_matrix[:, 1]) + 1
        # (37, 10)
        self._initialize(len(types), len(stages))
        for row in text_matrix:
            j, i, x = row
            self.matrix[i, j] = x
        self.datframe = pd.DataFrame(self.matrix, index=types, columns=stages)
        return self

    def to_csv(self, **kwargs):
        file = self.file.rsplit('.', 1)
        file[0] += '_matrix'
        self.datframe.to_csv('.'.join(file), **kwargs)


class SunburstStatic(object):

    ROOT_ATTR = {
        "name": 'Total',
        "color": '#fdf5e6',
    }

    def __init__(self, file):
        self.file = os.path.join(ROOT_, file)
        self.datframe = None
        self._numpy = None

    def as_datframe(self):
        if self.datframe is None:
            self.datframe = pd.read_table(self.file, index_col=0, sep='\t', header=0)
            self._numpy = self.datframe.to_numpy()
        return self

    def to_json(self, save=True, include_root=True, **kwargs):
        print("Data:", self._numpy, sep='\n')
        json_str = NodeArray().build_from_array(arr=self._numpy)\
            .to_json(include_root=include_root, **self.ROOT_ATTR)
        file = self.file.rsplit('.', 1)
        file[1] = 'json'
        if not save:
            return json_str
        with open('.'.join(file), 'w') as f:
            f.write(json_str)
            print('Done')


if __name__ == '__main__':
        # sc = StaticCellCounts('cells_counts.txt')
        # sc.build_dataframe().to_csv(header=True, index=True, sep='\t')
        ss = SunburstStatic('sunbusrt.txt').as_datframe()
        ss.to_json(include_root=False)

