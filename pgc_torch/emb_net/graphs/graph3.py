from pgc_torch.utils.graph import NLayerFeedForwardNet


class EMTABANNetGraph(NLayerFeedForwardNet):

    # hyper-parameters
    mask = 0.5      # dropout mask vector
    momentum = 0.5  # batch normalize momentum

    def forward(self, X, dim=-1):
        X = X.view(-1, 1*171*171)
        return super().forward(X, dim=-1)


class L4EMTABANNetGraph(EMTABANNetGraph):

    N_HIDDEN = 4
    HIDDEN_CELL = [1024, 512, 256, 15]


class L4EMTABANNetGraphV2(EMTABANNetGraph):

    N_HIDDEN = 4
    HIDDEN_CELL = [2048, 512, 256, 12]
    # HIDDEN_CELL = [4096, 512, 256, 12]


class L2EMTABANNetGraph(EMTABANNetGraph):

    N_HIDDEN = 2
    HIDDEN_CELL = [1024, 16]


class L5EMTABANNetGraph(EMTABANNetGraph):

    N_HIDDEN = 5
    HIDDEN_CELL = [1024, 512, 256, 128, 15]