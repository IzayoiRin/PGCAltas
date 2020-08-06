import copy
import torch as th
import torch.nn as nn

from pgc_torch.utils.graph import NLayerFeedForwardNet
from pgc_torch.utils.substructure import V2Inception, EMTABMetaClass, InceptionBaseRoute

IncMapping = dict()
EMTABMetaClass.IncMapping = IncMapping


class EMTABaseV2Inception(V2Inception, metaclass=EMTABMetaClass):
    __constants__ = ['inchanel', ]

    def __init__(self, inchanel):
        self.inchanel = inchanel
        super(EMTABaseV2Inception, self).__init__()
        self.assemble(inchanel)

    def extra_repr(self):
        return 'inchanel={}'.format(self.inchanel)


class EMTABV2Inception3a(EMTABaseV2Inception):
    N = 3
    chanelArr = [[32], [16, 16, 32], [8, 8, 8, 8, 16], [None, 16]]


class EMTABConvRoute(InceptionBaseRoute):

    Route = [
        ('conv', {'kernel': 7, 'stride': 2}),
        ('maxp', {'kernel_size': 3, 'stride': 2}),
        ('inc3a', dict()),
        ('maxp', {'kernel_size': 3, 'stride': 2}),
        ('avgp', {'kernel_size': 5}),
        ('fln', dict())
    ]

    Chanel = [32, None, 96, None, None, None]

    def __init__(self, dropout=False):
        self.FMapping = copy.deepcopy(self.FMapping)
        for k, v in IncMapping.items():
            self.FMapping[k] = v
        if dropout:
            self.FMapping['dp'] = nn.Dropout2d
            self.NON_WEIGHTS.append('dp')
        super(EMTABConvRoute, self).__init__()

    def cond_(self, func_name: str, dims):
        if func_name.startswith('inc'):
            return True
        else:
            return False

    def costume_assemble(self, func, dims, i, **p):
        return func(dims[i], **p)


class EMTABCNNetGraphLess(NLayerFeedForwardNet):

    inc = EMTABConvRoute(dropout=False)

    N_HIDDEN = 2
    HIDDEN_CELL = [1024, 32]

    __inchanel__ = 1
    __ifea__ = 171

    def add_inception(self):
        if self._ginit:
            print('Graph assembled, flush() first')
            return
        self.graph.add_module('v1inc', self.inc.assemble(self.__inchanel__))

    def assemble(self, ifea=521, ofea=4):
        self.add_inception()
        super(EMTABCNNetGraphLess, self).assemble(1536, ofea=ofea)
        return self

    def forward(self, X, dim=-1):
        flag = isinstance(X, th.Tensor) and len(X.size()) is 4 \
               and X.size(1) == self.__inchanel__ and X.size(-1) == X.size(-2) == self.__ifea__
        if not flag:
            raise ValueError('X must be shape like: [B, C, W, H] == [N, %s, %s, %s]'
                             % (self.__inchanel__, self.__ifea__, self.__ifea__))
        return super().forward(X, dim=-1)


if __name__ == '__main__':
    # g = EMTABCNNetGraphLess(dropout=False).assemble()
    g = EMTABConvRoute().assemble(1)
    # print(g.graph)
    e = th.randn(8, 1, 171, 171)
    e = g(e)
    print(e.size())
