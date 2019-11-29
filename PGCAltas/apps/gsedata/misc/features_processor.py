from PGCAltas.utils.StatExpr.StaUtills.FeaturesProcessor.preprocessors import FeaturesBasicPreProcessor, \
    FeaturesBasicScreenProcessor


class GSEBinDimensionPPFeatures(FeaturesBasicPreProcessor):

    def get_labels(self):
        label_dim = self.kwargs.get('dim')
        label_dim = 0 if label_dim == 'time' else 1
        return self.labels[label_dim]


class GSEBinDimensionSLTFeatures(FeaturesBasicScreenProcessor):

    def get_labels(self):
        label_dim = self.kwargs.get('dim')
        label_dim = 0 if label_dim == 'time' else 1
        return self.labels[label_dim]
