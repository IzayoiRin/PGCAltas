"""
mnist dataset
"""
import numpy as np
import torch as th
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize

from pgc_torch.utils.graph import NLayerFeedForwardNet
from pgc_torch.utils.models import PrettyFeedForward


class MnistDataLoader(object):

    def __init__(self, train):
        self.train = train
        self.raw_data = self.mnist_dataset()

    def mnist_dataset(self):
        """
        mnist = MNIST(root='./mnist/', train=True)
        print(mnist[0], len(mnist))
        print(mnist[0][0].show())
        :param train:
        :return:
        """
        func = Compose(
            [
                ToTensor(),
                Normalize(mean=(0.1307,), std=(0.3081,))
            ])
        return MNIST(root='./mnist/', train=self.train, transform=func)

    def __call__(self, batch_size=128, shuffle=True, split=False, size=0.1):
        if not split:
            return DataLoader(dataset=self.raw_data, batch_size=batch_size, shuffle=shuffle)
        return [DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle) for data in self._split_t_v(size)]

    def _split_t_v(self, size):
        total = len(self.raw_data)
        threshold = np.floor(total * size).astype(int)
        idx = range(total)
        vidx = np.random.choice(idx, threshold, replace=False)
        tidx = np.setdiff1d(idx, vidx)
        return [self.raw_data[i] for i in tidx], [self.raw_data[i] for i in vidx]


class MnistNetGraph(NLayerFeedForwardNet):

    N_HIDDEN = 5
    HIDDEN_CELL = [500, 300, 100, 40, 15]

    # hyper-parameters
    mask = 0.5      # dropout mask vector
    momentum = 0.5  # batch normalize momentum

    def forward(self, X, dim=-1):
        X = X.view(-1, 1*28*28)
        return super().forward(X, dim=-1)


class MnistNet(PrettyFeedForward):

    data_loader_class = MnistDataLoader
    model_graph_class = MnistNetGraph
    optimizer_class = th.optim.Adam
    loss_func_class = th.nn.NLLLoss

    loader_params = {
        "train": {
            "batch_size": 128,
            "shuffle": True,
        },
        "test": {
            "batch_size": 128,
            "shuffle": False,
        }
    }

    # hyper-parameters
    lr = 1e-3       # learning rate
    l1_lambda = 0.5     # l1-penalty coef
    l2_lambda = 0.01    # l2-penalty coef
    step = 2    # measure_progress step k
    patient = 3     # early stopping patient
    alpha = 1e-2    # early stopping threshold

    def __init__(self, *args, **kwargs):
        super(MnistNet, self).__init__(*args, **kwargs)
        self._acc = list()

    def get_data_loader(self, train):
        p = self.loader_params['train'] if train else self.loader_params['test']
        if train:
            p["split"] = True
            p["size"] = 0.1
        return self.data_loader_class(train=train)(**p)

    @property
    def acc(self):
        return np.mean(self._acc)

    def analysis(self, label, ypre, preP):
        self._acc.append(self.accuracy(ypre, label).item())

    def eval_batch(self, dl):
        super(MnistNet, self).eval_batch(dl)
        print('Average Accuracy: %s' % self.acc)


if __name__ == '__main__':
    net = MnistNet(1*28*28, 10, reg=None, batch_nor=True)
    net.train(20, 'GL')
    # m = MnistDataLoader(True).raw_data
    # print(m[0][0].size())