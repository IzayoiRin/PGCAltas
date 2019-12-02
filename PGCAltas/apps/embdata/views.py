from django.shortcuts import render

# Create your views here.
from embdata.misc.features_select import EMBTABinomalEIMProcess, EMBTABinomalEIMAnalysis


def features_select():
    eim = EMBTABinomalEIMProcess(r"[A-Za-z]+2_Expr.*")
    # eim.flushed()
    eim.test_size = 0.7
    eim.execute_eim_process()

    aly = EMBTABinomalEIMAnalysis()
    aly.execute_eim_analysis("trans_and_sig", "acc_between_select")


features_select()
