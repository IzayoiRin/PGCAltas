from embdata.misc.features_project import EMBTABinomalEIMProcess, EMBTABinomalEIMAnalysis, EMBTABinomalDimensionEstimate


class FeatuersProjectWorkFlow(object):

    def __init__(self, file_name):
        self.file_name = file_name
        self.flush = False
        self.test_size = 0.7
        self.eim_choice = list()

    def eim(self):
        eim = EMBTABinomalEIMProcess(self.file_name)
        # dataReader Flushed
        if self.flush:
            eim.flushed()

        eim.test_size = self.test_size
        # Calculate Equ_importance Integral Matrix
        eim.execute_eim_process()
        return self

    def eim_analysis(self):
        # Get Analysis Handler
        aly = EMBTABinomalEIMAnalysis()
        # Get Selected Expression Matrix and Significant Score Matrix
        # Get the Accuracy Matrix between Raw Expression and Selected Expression Matrix
        aly.execute_eim_analysis(*self.eim_choice)
        return self

    def estimate(self, **kwargs):
        fext = EMBTABinomalDimensionEstimate()
        fext.test_size = kwargs.pop('test_size', None) or self.test_size
        fext.execute_estimate_process(**kwargs)


def features(flush=False, **kwargs):
    workflow = FeatuersProjectWorkFlow(r"[A-Za-z]+2_Expr.*")
    workflow.flush = flush
    workflow.test_size = 0.7
    workflow.eim_choice = ["trans_and_sig", "acc_between_select"]
    # n_components=12, after_filter=120, barnes_hut=0.5, test_size=0.4
    workflow.eim().eim_analysis().estimate(**kwargs)
