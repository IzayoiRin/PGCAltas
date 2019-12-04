import os

# from django.http import HttpResponse
# from django.shortcuts import render
# from rest_framework.generics import GenericAPIView
# from rest_framework.response import Response

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


def features(flush=False):
    workflow = FeatuersProjectWorkFlow(r"[A-Za-z]+2_Expr.*")
    workflow.flush = flush
    workflow.test_size = 0.7
    workflow.eim_choice = ["trans_and_sig", "acc_between_select"]

    workflow.eim().eim_analysis().estimate(n_components=10, after_filter=120)


features(True)

# class FeaturesScreenAPIView(GenericAPIView):
#
#     def get(self, request):
#         flush = request.query_params.get('flush', None)
#         features_screen(flush)
#         eim_mtx_file = os.path.join(__DATA_ROOT__, 'texts', 'RDFBinomialFlow.txt')
#         with open(eim_mtx_file, 'r', encoding='utf-8') as f:
#             text = f.read()
#         return HttpResponse(text, content_type="text/plain")
