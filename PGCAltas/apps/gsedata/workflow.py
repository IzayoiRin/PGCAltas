from gsedata.misc.features_select import GSEBinDimensionEIMProcess, GSEBinDimensionEIMAnalysis


class FeatuersProjectWorkFlow(object):

    def __init__(self, file_name):
        self.file_name = file_name
        self.flush = False
        self.test_size = 0.7
        self.eim_choice = list()

    def eim(self):
        eim = GSEBinDimensionEIMProcess(self.file_name)
        # dataReader Flushed
        if self.flush:
            eim.flushed()

        eim.test_size = self.test_size
        # Calculate Equ_importance Integral Matrix
        eim.execute_eim_process()
        return self

    def eim_analysis(self):
        # Get Analysis Handler
        aly = GSEBinDimensionEIMAnalysis()
        # Get Selected Expression Matrix and Significant Score Matrix
        # Get the Accuracy Matrix between Raw Expression and Selected Expression Matrix
        aly.execute_eim_analysis(*self.eim_choice)
        return self


def features(flush=False, **kwargs):
    workflow = FeatuersProjectWorkFlow(r"^E.*")
    workflow.flush = flush
    workflow.test_size = 0.7
    workflow.eim_choice = ["trans_and_sig", "acc_between_select"]
    workflow.eim().eim_analysis()
