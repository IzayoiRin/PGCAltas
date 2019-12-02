"""
CONSTANT for calculating IMPORTANCE AREA
"""

from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "EMTAB6967"
FILE_TYPE = r'\.txt'
PKL_FILE = 'OBJEMTAB696719120212.pkl'
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
