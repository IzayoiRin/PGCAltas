import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader

from PGCAltas.utils.statUniversal import train_test_split
from pgc_torch.utils.data import EMTABaseDataset


class Data(Dataset):

    def __init__(self, x, y):
        self._dataset = x
        self._label = y

    def __getitem__(self, item):
        return self._dataset[item], self._label[item]

    def __len__(self):
        return self._label.size(0)

    def __str__(self):
        return "WPGCsTensor[N%s, C%s*W%s*H%s]" % self._dataset.size()


class WholePGCsDatasetFactory(EMTABaseDataset):

    raw_pklfile = 'WholePGCs.pkl'

    te_rate = 0.2
    va_rate = 0.3

    __chanel__ = 1
    __ifea__ = 171
    __non_stage__ = ["mixed_gastrulation", 'E6.5', ]
    __stage_dict__ = {
        ('E6.75', 'E7.0', 'E7.25', 'E7.5'): 0,
        ('E7.75', 'E8.0', 'E8.25', 'E8.5'): 1,
    }

    def __init__(self):

        self.samples_ = None
        self.stages_ = None
        self._ds_list = None

        super(WholePGCsDatasetFactory, self).__init__(self.raw_pklfile)

    def resolute_dat_(self, **kwargs):
        self.samples_ = self.rdata.loc[:, 'cid'].to_numpy(np.str)  # shape: (424,)
        self.stages_ = self.rdata.loc[:, 'stage'].to_numpy(np.str)  # shape: (424, )

        nsid = np.hstack([np.argwhere(self.stages_ == ns).reshape(-1) for ns in self.__non_stage__])
        sidx = np.setdiff1d(np.arange(len(self.stages_)), nsid)
        col = list(range(self.rdata.shape[1]))
        col.pop(1)
        expr = self.rdata.iloc[sidx, col].to_numpy()  # shape: (424, 29241)
        stage = self.stages_[sidx, ]
        xtr, xte, ytr, yte = train_test_split(mode='L')(expr, stage, self.te_rate)
        xtr, xva, ytr, yva = train_test_split(mode='L')(xtr, ytr, self.va_rate)
        self.samples_ = [xte[:, 0], xtr[:, 0], xva[:, 0]]
        xte, xtr, xva = xte[:, 1:], xtr[:, 1:], xva[:, 1:],
        self._ds_list = [(xte, yte), (xtr, ytr), (xva, yva)]

    def trans2tensor(self, x, y):
        x = x.astype(np.float32)
        x = th.tensor(x, dtype=th.float32)
        x = x.view(x.size(0), self.__chanel__, self.__ifea__, self.__ifea__)
        y = th.tensor(list(map(self.stage2label, y)))
        return x, y

    def stage2label(self, stage):
        for k, v in self.__stage_dict__.items():
            if stage in k:
                return v

    def call_standard_train(self):
        (xtr, ytr), (xva, yva) = self._ds_list[1], self._ds_list[-1]
        xtr = np.vstack([xtr, xva])
        ytr = np.hstack([ytr, yva])
        return Data(*self.trans2tensor(xtr, ytr))

    def __call__(self, train):
        return Data(*self.trans2tensor(*self._ds_list[train]))


class WholePGCsDataloader(object):

    dataset = WholePGCsDatasetFactory()

    def __init__(self, train):
        self.train = train

    def get_dataset(self):
        return self.dataset

    def __call__(self, batch_size=128, shuffle=True, split=False):
        """
        train 0 -----------> (dataset(0))
            split 1 train 0 ---> (dataset(0))
            split 0 train 0 ---> (dataset(0))

        train 1
            split 1 train 1 ---> (dataset(1), dataset(-1))
            split 0 train 1 ---> concat(dataset(1), dataset(-1))
        """
        ds_cls = self.get_dataset()
        if not self.train:
            return DataLoader(dataset=ds_cls(bool(self.train)), batch_size=batch_size, shuffle=shuffle)
        if split:
            dat = [ds_cls(train=1), ds_cls(train=-1)]
            return [DataLoader(dataset=d, batch_size=batch_size, shuffle=shuffle) for d in dat]
        return DataLoader(dataset=ds_cls.call_standard_train(), batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    w1, w2 = WholePGCsDataloader(True)(70, split=True)
    # w3 = WholePGCsDataloader(False)(128)
    # w4 = WholePGCsDataloader(True)(128, split=False)
    # for _ in range(3):
    #     for i, j in w1:
    #         j = j  # type: th.Tensor
    #         print(j.unique())
    #         print(j[:5])
    #     print()
    # for i, j in w2:
    #     j = j  # type: th.Tensor
    #     print(j.unique())
    # print()
    # for i, j in w3:
    #     j = j  # type: th.Tensor
    #     print(j.unique())
