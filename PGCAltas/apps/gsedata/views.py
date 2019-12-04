from django.shortcuts import render

# Create your views here.

from gsedata import patch_const
patch_const()
from gsedata.misc.features_select import GSEBinDimensionEIMProcess, GSEBinDimensionEIMAnalysis


def features_screen(flush=False):
    eim = GSEBinDimensionEIMProcess(r"^E.*")
    if flush:
        # dataReader Flushed
        eim.flushed()
    # Calculate Equ_importance Integral Matrix
    eim.execute_eim_process()
    # Get Analysis Handler
    aly = GSEBinDimensionEIMAnalysis()
    anlysis_handler = aly.execute_eim_analysis
    # Get Selected Expression Matrix and Significant Score Matrix
    # Get the Accuracy Matrix between Raw Expression and Selected Expression Matrix
    anlysis_handler("trans_and_sig", "acc_between_select")


features_screen(True)
