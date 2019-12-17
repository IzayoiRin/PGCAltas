"""
CONSTANT for calculating IMPORTANCE AREA
"""

from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "EMTAB6967"
FILE_TYPE = r'\.txt'
# SMALL DATASET
# PKL_FILE = 'OBJEMTAB696719121616_small.pkl'
# LARGE DATASET
PKL_FILE = 'OBJEMTAB696719121710_large.pkl'
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
DIMENSION ESTIMATOR MODEL PARAMS
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

# 2D-Viewer params
TSNE_PARAMS = {
    'perplexity': 30.0,
    'learning_rate': 200.0,
    'n_iter': 1000,
    'n_iter_without_progress': 50 * 6,
    'random_state': 0,
    'method': 'exact',
    # 'angle': 0.5 for 'method': 'barnes_hut'
}

SPSVD_PARAMS = {
    'algorithm': 'randomized',
    'n_iter': 7,
    'random_state': 0,
}


"""
CONSTANT for SVM Classify
"""
ESTIMATED_EXPR_PKL = {
    "binomial": "EstimatedBinomialFlow.pkl",
}


"""
SUPPORT VECTOR MACHINE PARAMS
"""
SVM_PARAMS = {
    'C': 1.0,
    # linear, poly, rbf, sigmoid, precomputed
    'kernel': 'rbf',
    # max level power of kernel ploy
    'degree': 3,
    # coef of kernel function
    'gamma': 'scale',
    # bias of kernel ploy and sigmoid
    'coef0': 0.0,
    'max_iter': -1,
    'shrinking': True,
    'probability': False,
    # multi classify decision: ovr, crammer_singer
    # ovr: 将待分类中的某一类当作正类，其他全部归为负类，通过这样求取得到每个类别作为正类时的正确率，取正确率最高的那个类别为正类
    # ovo: 目标函数设置多个参数值，最后进行优化，得到不同类别的参数值大小
    'decision_function_shape': 'ovr',
    'random_state': 0,
    'cache_size': 200
}
