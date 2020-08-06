import numpy as np
import torch as th


device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')


def tensor_exp():
    arr = np.arange(10)
    t1 = th.tensor(arr)
    print('th.arr', t1)
    t2 = th.tensor(arr.reshape(2, -1))
    print('th.mat', t2)
    zero = th.zeros((2, 2))
    print('th.zero', zero)
    one = th.ones_like(t1)
    print('th.one', one)
    np_t1 = t1.numpy()
    print('to_numpy', np_t1)
    size = t2.size()
    print('size', size)
    view = t1.view(5, -1)
    print('reshape', view)
    trans = view.transpose(0, 1)
    print('T', trans)

    print('t1 type', t1.dtype)

    dtypes = {
        'b32f': [th.float32, th.float],
        'b64f': [th.float64, th.double],
        'b16f': [th.float16, th.half],
        'b8i': th.int8,
        'ub8i': th.uint8,
        'b16i': [th.int16, th.short],
        'b32i': [th.int32, th.int],
        'b64i': [th.int64, th.long]
    }

    print('to float', t1.float().dtype)

    one2 = th.ones_like(t2)
    print(one2.add(1), one2)
    one2.add_(1)
    print(one2)

    cudat1 = t1.to(device)
    print('cuda', cudat1)

    x = th.ones_like(zero, dtype=th.float, requires_grad=True)
    y = x.matmul(th.tensor([[0.2], [0.1]]))
    print('pic', y.grad_fn)

    s = th.mean(y)
    s.backward(retain_graph=False)
    print('gradient', x.grad)
    print('x', x.detach())

    with th.no_grad():
        z = x.matmul(th.tensor([[0.2], [0.1]]))
        print(z.requires_grad)
    z.requires_grad_(True)
    print(z.requires_grad)

    print(t2.max(dim=-1))


def linear():
    # set
    x = th.randn(100, 1)
    # y = 0.5 * X + 0.2
    y = x.matmul(th.tensor([[0.5]])) + 0.2

    # init varibles
    weights = th.randn(1, 1, requires_grad=True)
    bias = th.randn(1, requires_grad=True)

    for _ in range(100):
        ypre = x.matmul(weights) + bias

        # MSE LOSS
        lr = 0.1
        loss = th.pow(ypre - y, 2).mean()

        # zerolize variable gradient
        for v in [weights, bias]:
            if v.grad is not None:
                v.grad.data.zero_()

        # backward
        loss.backward()

        # update
        weights.data -= lr * weights.grad
        bias.data -= lr * bias.grad

    print(weights.detach().item(), bias.detach().item())


from torch import nn


class Linear(nn.Module):
    """Model Calculation Pic"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        """forward calculation"""
        return self.linear(x)


def linear_api():
    X = th.randn(100, 1).to(device)
    y = X.matmul(th.tensor([[0.5]], device=device)) + 0.2

    model = Linear().to(device)

    # optimizer
    from torch import optim
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    loss_fn = nn.MSELoss()

    for _ in range(500):
        # zerolize optimizer
        optimizer.zero_grad()

        # predict
        out = model(X)

        # cal loss
        loss = loss_fn(out, y)

        # backward
        loss.backward()

        # update variables
        optimizer.step()

    for v in model.parameters():
        print(v.detach().to('cpu', dtype=th.float32))


from torch.utils.data import Dataset


class CDataset(Dataset):

    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


class ANNUniModel(nn.Module):
    """
    2 ful combo layer, activated by RELU, out by log_softmax
    """

    activator = nn.functional.relu
    out_func = nn.functional.log_softmax

    def __init__(self, infactors=10):
        super(ANNUniModel, self).__init__()
        # first ful-comb layer, infactors = R[n], outfactors = 100
        self.fc1 = nn.Linear(infactors, 100)
        # second ful-comb layer, infactors = 100, outfactors = 10
        self.fc2 = nn.Linear(100, 10)

    def forward(self, X):
        # first ful-comb layer out, R[100], relu activated
        fc1out = self.activator(self.fc1(X))
        # second ful-comb layer out, R[10], relu activated
        fc2out = self.activator(self.fc2(fc1out))
        # log(P(x))
        return self.out_func(fc2out)


class NetWorkFlow(object):

    dataloader_class = None
    loss_func = nn.functional.nll_loss

    DEVICE = device
    LR = 1e-3

    def __init__(self):
        self.model = ANNUniModel()
        if self.DEVICE:
            self.model = self.model.to(self.DEVICE)

        self.optimizer = th.optim.Adam(self.model.parameters(), lr=self.LR)

    def get_dataloader(self, train):
        return self.dataloader_class(train=train)

    def get_loss(self, x, y):
        return self.loss_func(x, y)

    def one_train(self):
        tr_dataloader = self.get_dataloader(True)
        loss_arr = list()
        for batch, (x, label) in enumerate(tr_dataloader):
            if self.DEVICE:
                x = x.to(self.DEVICE)
                label = label.to(self.DEVICE)
            # init optimizer
            self.optimizer.zero_grad()
            # forward prediction
            logP = self.model(x)
            # Cross Entropy Loss: -sigma(Y*log(P))
            loss = self.get_loss(label, logP)
            # gradient backward
            loss.backward()
            # update parameters
            self.optimizer.step()
            # cal average loss
            loss_arr.append(loss)
            ave_loss = np.mean(loss_arr)

    def train(self, iter_n):
        for i in range(iter_n):
            # iteration training
            self.one_train()
            path_m = "%s"
            path_o = "%s"
            self.save(self.model, path_m)
            self.save(self.optimizer, path_o)

    def eval(self):
        te_dataloader = self.get_dataloader(False)
        loss_arr = list()
        with th.no_grad():
            for batch, (x, label) in enumerate(te_dataloader):
                if self.DEVICE:
                    x = x.to(self.DEVICE)
                    label = label.to(self.DEVICE)

                logP = self.model(x)
                loss = self.get_loss(label, logP)
                loss_arr.append(loss)

                preP, ypre = logP.max(dim=-1)

                acc = ypre.eq(label).float().mean()

    def save(self, obj, path):
        th.save(obj.state_dict, path)

    def load(self, path):
        self.model.load_state_dict(th.load(path))
        self.optimizer.load_state_dict(th.load(path))


if __name__ == '__main__':
    tensor_exp()
    # linear()
    # linear_api()

