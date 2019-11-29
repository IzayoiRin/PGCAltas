from django.shortcuts import render

# Create your views here.
from embdata.misc.features_select import EMBTABinomalEIMProcess


def features_select():
    eim = EMBTABinomalEIMProcess(r"[A-Za-z]+2_Expr.*")
    eim.flushed()
    eim.execute_eim_process()


features_select()
