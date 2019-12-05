from rest_framework.generics import GenericAPIView
from rest_framework.response import Response

from embdata import patch_const
patch_const()

from embdata.misc.features_project import EMBTABinomalEIMProcess, EMBTABinomalEIMAnalysis, EMBTABinomalDimensionEstimate

__DATA_ROOT__ = 'dataset\EMTAB6967'


class FeatuersProjectWorkFlow(object):

    def __init__(self, file_name):
        self.file_name = file_name
        self.flush = False
        self.test_size = 0.0
        self.eim_choice = list()

    def eim(self):
        eim = EMBTABinomalEIMProcess(self.file_name)

        if self.flush:
            eim.flushed()

        eim.test_size = self.test_size
        eim.execute_eim_process()
        return self

    def eim_analysis(self):
        aly = EMBTABinomalEIMAnalysis()
        aly.execute_eim_analysis(*self.eim_choice)
        return self

    def estimate(self, **kwargs):
        fext = EMBTABinomalDimensionEstimate()
        fext.execute_estimate_process(**kwargs)


def features(flush=False, **kwargs):
    workflow = FeatuersProjectWorkFlow(r"[A-Za-z]+2_Expr.*")
    workflow.flush = flush
    workflow.test_size = 0.7
    workflow.eim_choice = ["trans_and_sig", "acc_between_select"]
    # n_components=12, after_filter=120, barnes_hut=0.5
    workflow.eim().eim_analysis().estimate(**kwargs)


class FeaturesScreenAPIView(GenericAPIView):

    def get(self, request):
        query_dict = request.query_params
        # TODO: Serializer optim
        query = {k: int(v) for k, v in query_dict.items()}
        print(query)
        features(**query)
        # eim_mtx_file = os.path.join(__DATA_ROOT__, 'texts', 'RDFBinomialFlow.txt')
        # with open(eim_mtx_file, 'r', encoding='utf-8') as f:
        #     text = f.read()
        # return HttpResponse(text, content_type="text/plain")
        return Response({"msg": "Finish"})
