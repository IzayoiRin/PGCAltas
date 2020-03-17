import os
import numpy as np
import pandas as pd
import torch as th

from pgc_torch.emb_net.dataloader import WholePGCsDataloader
from pgc_torch.emb_net.graphs.graph import EMTABCNNetGraph as G1
from pgc_torch.emb_net.graphs.graph0 import EMTABCNNetGraph as G0
from pgc_torch.emb_net.graphs.graph2 import EMTABCNNetGraphLess as G2
from pgc_torch.emb_net.graphs.graph3 import L2EMTABANNetGraph as G3
from pgc_torch.emb_net.graphs.graph3 import L4EMTABANNetGraph as G4
from pgc_torch.emb_net.graphs.graph3 import L4EMTABANNetGraphV2 as G6
from pgc_torch.emb_net.graphs.graph3 import L5EMTABANNetGraph as G5
from pgc_torch.utils.models import PrettyFeedForward


class EMTABNet(PrettyFeedForward):

    data_loader_class = WholePGCsDataloader
    # model_graph_class = ANNetGraph
    model_graph_class = G1
    optimizer_class = th.optim.Adam
    loss_func_class = th.nn.NLLLoss

    loader_params = {
        "train": {
            "batch_size": 70,
            "shuffle": True,
        },
        "test": {
            "batch_size": 100,
            "shuffle": False,
        }
    }

    # hyper-parameters
    lr = 1e-3  # learning rate
    l1_lambda = 0.5  # l1-penalty coef
    l2_lambda = 0.01  # l2-penalty coef
    step = 10  # measure_progress step k
    patient = 3  # early stopping patient
    alpha = 0.5  # early stopping threshold

    def __init__(self, ofea, **kwargs):
        super(EMTABNet, self).__init__(ifea=171 * 171, ofea=ofea, **kwargs)
        self.CHECK_POINT = 'cp{}ep%s.tar'.format(self.model_graph_class.__name__)
        self._acc = list()
        self.acc_curve = list()
        self._loss = list()
        self.vloss_curve = list()
        self.tloss_curve = list()

        self.eval_ret = list()
        self.pre_accuracy = None

        self.samples_ = self.data_loader_class.dataset.samples_

    def get_data_loader(self, train):
        p = self.loader_params['train'] if train else self.loader_params['test']
        if train:
            p["split"] = True
        return self.data_loader_class(train=train)(**p)

    @property
    def epoch_acc(self):
        return np.mean(self._acc)

    @property
    def epoch_loss(self):
        return np.mean(self._loss)

    def analysis(self, label, ypre, preP):
        """

        :param label: size(batch) true class
        :param ypre: size(batch) pre class
        :param preP: size(batch) pre prob
        :return:
        """
        self._acc.append(self.accuracy(ypre, label).item())
        if not getattr(self, 'validate', False):
            self.eval_ret.append(th.stack([label.float(), ypre.float(), preP], dim=1))

    def train_batch(self, dl):
        super(EMTABNet, self).train_batch(dl)
        self.tloss_curve.append(self.epoch_loss)

    def eval_batch(self, dl):
        self._acc = list()
        super(EMTABNet, self).eval_batch(dl)
        print('Average Accuracy: %s' % self.epoch_acc)
        if getattr(self, 'validate', False):
            self.acc_curve.append(self.epoch_acc)
            self.vloss_curve.append(self.epoch_loss)
        else:
            ret = th.cat(self.eval_ret, dim=0)
            self.pre_accuracy = self.accuracy(ret[0], ret[1])
            path = os.path.join(self.csv_path, 'EvalCurves%s.txt' % self.model_graph_class.__name__)
            pd.DataFrame(np.hstack([self.samples_[0].reshape(-1, 1), ret.cpu().numpy()]),
                         columns=['cids', 'label', 'predict', 'prob'])\
                .to_csv(path, sep='\t', index=True, header=True)

    def model_persistence(self):
        super(EMTABNet, self).model_persistence()
        curves = {
            "Accaracy": self.acc_curve,
            "TrLoss": self.tloss_curve,
            "VaLoss": self.vloss_curve
        }
        path = os.path.join(self.csv_path, 'EpochCurves%s.txt' % self.model_graph_class.__name__)
        df = pd.DataFrame(curves.values()).T
        df.columns = curves.keys()
        df.to_csv(path, sep='\t', index=True, header=True)


def main():
    Net = EMTABNet
    # G = [G1, G2, G3, G4]
    G = [G0]
    # p = [(0.5, 200, 10, 2), (0.1, 200, 10, 2), (0.08, 200, 5, 2), (0.1, 200, 5, 2)]
    p = [(0.5, 1, 10, 0)]
    for g, (aph, n, stp, cp) in zip(G, p):
        Net.model_graph_class = g
        Net.alpha = aph
        Net.step = stp
        net1 = Net(2, reg=None, dropout=False)
        net1.train(n, 'PQ', checkpoint=cp)


def test():
    Net = EMTABNet
    G = [G5]
    p = [(0.5, 5, 10, '0319cpL5EMTABANNetGraphep188.tar')]
    for g, (aph, n, stp, cp) in zip(G, p):
        Net.model_graph_class = g
        Net.alpha = aph
        Net.step = stp

        net2 = Net(2, reg=None, dropout=False)
        try:
            net2.load(ckpath=cp)
        except Exception as e:
            return
        # net2.train(n, 'PQ', checkpoint=1)
        net2.eval()


if __name__ == '__main__':
    # main()
    test()
