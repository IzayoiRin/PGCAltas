import logging

import copy
import torch as th
import torch.nn as nn

from pgc_torch.utils.graph import NLayerFeedForwardNet
from pgc_torch.utils.substructure import V1Inception, InceptionBaseRoute, EMTABMetaClass

logger = logging.getLogger('django')

IncMapping = dict()
EMTABMetaClass.IncMapping = IncMapping


class EMTABaseV1Inception(V1Inception, metaclass=EMTABMetaClass):
    __constants__ = ['inchanel', ]

    def __init__(self, inchanel):
        self.inchanel = inchanel
        super(EMTABaseV1Inception, self).__init__()
        self.assemble(inchanel)

    def extra_repr(self):
        return 'inchanel={}'.format(self.inchanel)


class EMTABV1Inception3a(EMTABaseV1Inception):
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[64], [96, 128], [16, 32], [None, 32]]


class EMTABV1Inception3b(EMTABaseV1Inception):
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[128], [128, 192], [32, 96], [None, 64]]


class EMTABV1Inception4a(EMTABaseV1Inception):
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[192], [96, 208], [16, 48], [None, 64]]


class EMTABV1Inception4b(EMTABaseV1Inception):
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[160], [112, 224], [24, 64], [None, 64]]


class EMTABV1Inception4c(EMTABaseV1Inception):
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[128], [128, 256], [24, 64], [None, 64]]


class EMTABV1Inception4d(EMTABaseV1Inception):
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[112], [144, 288], [32, 64], [None, 64]]


class EMTABV1Inception4e(EMTABaseV1Inception):
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[256], [160, 320], [32, 128], [None, 128]]


class EMTABV1Inception5a(EMTABaseV1Inception):
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[256], [160, 320], [32, 128], [None, 128]]


class EMTABV1Inception5b(EMTABaseV1Inception):
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[384], [192, 384], [48, 128], [None, 128]]


class EMTABConvRoute(InceptionBaseRoute):
    """
        a = EMTABConvRoute().assemble(1)
        print(a.graph)
        e = th.randn(8, 1, 171, 171)
        print("input: batch %s, C%s, W%s, H%s" % e.size())
        e = a(e)
        print("output: batch %s, C%s, W%s, H%s" % e.size())
    """

    Route = [
        ('conv', {'kernel': 7, 'stride': 2}),
        ('maxp', {'kernel_size': 3, 'stride': 2}),
        ('conv', {'kernel': 1}),
        ('conv', {'kernel': 3, 'padding': 1}),
        ('maxp', {'kernel_size': 3, 'stride': 2}),
        ('inc3a', dict()),
        ('inc3b', dict()),
        ('maxp', {'kernel_size': 3, 'stride': 2, 'padding': 1}),
        ('inc4a', dict()),
        ('inc4b', dict()),
        ('inc4c', dict()),
        ('inc4d', dict()),
        ('inc4e', dict()),
        ('maxp', {'kernel_size': 3, 'stride': 2, 'padding': 1}),
        ('inc5a', dict()),
        ('inc5b', dict()),
        ('avgp', {'kernel_size': 5}),
        ('fln', dict())
    ]

    Chanel = [64, None, 64, 192, None, 256, 480, None, 512, 512, 512, 528, 832, None, 832, 1024, None, None]

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


class EMTABCNNetGraph(NLayerFeedForwardNet):

    inc = EMTABConvRoute(dropout=False)

    # N_HIDDEN = 2
    N_HIDDEN = 3
    # HIDDEN_CELL = [256, 32]
    HIDDEN_CELL = [1000, 256, 12]

    __inchanel__ = 1
    __ifea__ = 171

    def add_inception(self):
        if self._ginit:
            print('Graph assembled, flush() first')
            return
        self.graph.add_module('v1inc', self.inc.assemble(self.__inchanel__))

    def assemble(self, ifea=521, ofea=3):
        self.add_inception()
        super(EMTABCNNetGraph, self).assemble(1024, ofea=ofea)
        return self

    def forward(self, X, dim=-1):
        flag = isinstance(X, th.Tensor) and len(X.size()) is 4 \
               and X.size(1) == self.__inchanel__ and X.size(-1) == X.size(-2) == self.__ifea__
        if not flag:
            raise ValueError('X must be shape like: [B, C, W, H] == [N, %s, %s, %s]'
                             % (self.__inchanel__, self.__ifea__, self.__ifea__))
        return super().forward(X, dim=-1)


if __name__ == '__main__':
    g = EMTABCNNetGraph(dropout=True).assemble()
    e = th.randn(8, 1, 171, 171)
    e = g(e)
    print(e.size())