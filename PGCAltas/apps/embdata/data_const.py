"""
CONSTANT for calculating IMPORTANCE AREA
"""

from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "EMTAB6967"
FILE_TYPE = r'\.txt'
PKL_FILE = 'OBJEMTAB696719120413.pkl'
DIMENSIONS = ['binomial', ]

NEG = r"^.*NEG"
POS = r"^.*POS"

"""
CONSTANT for analysis IMPORTANCE AREA
"""

IMPKL = {
    'binomial': "RDFBinomialFlow.pkl",
}
FIT_PKL = {
    'binomial': "RDFBinomialClassifier.pkl:SigScoreBinomialFlow.pkl",
}
CLASSIFIER_MODEL = RandomForestClassifier
THRESHOLD = 0.6


"""
RANDOM FOREST CLASSIFIER PARAMS
"""
RDF_PARAMS = {
    'n_estimators': 1000,
    'random_state': 0,
    'n_jobs': -1,
    'oob_score': True
}


"""
CONSTANT for Reduce Dimension of Expression Matrix Features
"""
SIGEXPR_PKL = {
    "binomial": "ExprBinomialFlow.pkl",
}


"""
REDUCE DIMENSION MODEL PARAMS
"""
LDA_PARAMS = {
    # svd , lsqr, eigen
    'solver': 'svd',
    # [0, 1] or auto
    'shrinkage': None,
    'priors': None,
}

PCA_PARAMS = {
    'random_state': 0,
}

FILTER_RATES = 0.2
COMPONENTS_RATES = 0.75
