from PGCAltas.utils.StatExpr.StatProcessor.FeaturesProcessor.processors import FeaturesBasicPreProcessor, \
    FeaturesBasicScreenProcessor


class GSEBinDimensionPPFeatures(FeaturesBasicPreProcessor):

    def get_labels(self):
        label_dim = self.kwargs.get('dim')
        label_dim = 0 if label_dim == 'time' else 1
        return self.labels[label_dim]

    def fit_encode(self, fit, *args, **kwargs):
        self.labels = [fit(ls.reshape(-1, 1), *args, **kwargs).toarray()
                       for ls in self.labels]
        return self


class GSEBinDimensionSLTFeatures(FeaturesBasicScreenProcessor):

    def get_labels(self):
        label_dim = self.kwargs.get('dim')
        label_dim = 0 if label_dim == 'time' else 1
        return self.labels[label_dim]
