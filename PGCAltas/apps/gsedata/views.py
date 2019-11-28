from django.shortcuts import render

# Create your views here.
from PGCAltas.utils.StatExpr import cal_imp_area as cal
from PGCAltas.utils.StatExpr import analysis_imp as aly


def features_select():
    # dataReader Flushed
    cal.__READER_FLUSHED = False
    # Calculate Equ_importance Integral Matrix
    cal.execute_eim_process()
    # Get Analysis Handler
    anlysis_handler = aly.execute_eim_analysis
    # Get Selected Expression Matrix and Significant Score Matrix
    # Get the Accuracy Matrix between Raw Expression and Selected Expression Matrix
    anlysis_handler("trans_and_sig", "acc_between_select")


features_select()
