"""
CONSTANT for calculating IMPORTANCE AREA
"""
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = 'GSE120963'
FILE_TYPE = r'\.txt'
PKL_FILE = 'OBJGSE12096319112916.pkl'
DIMENSIONS = ['time', 'loc']


"""
CONSTANT for analysis IMPORTANCE AREA
"""

IMPKL = {
    'time': "RDFTimeFlow.pkl",
    'loc': "RDFLocFlow.pkl"
}
FIT_PKL = {
    'loc': 'RDFLocClassifier.pkl:SigScoreLocFlow.pkl',
    'time': 'RDFTimeClassifier.pkl:SigScoreTimeFlow.pkl'
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
