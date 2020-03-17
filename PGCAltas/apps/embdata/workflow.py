from embdata.misc.features_project import EMBTABinomalEIMProcess, EMBTABinomalEIMAnalysis, EMBTABinomalDimensionEstimate
from embdata.misc.reader import BinomialPredictDataReader
from embdata.misc.svm_classifier import EMBTASVMProcess
from embdata.misc.validator import EMBTABinomalValidator


"""
workflow:

    // FeaturesProjectWorkFlow //
    
    1. pre-process
        data_reader built
        PKL_FILE pointed to built reader
        dumps pre-processed data
        * dumps RDF classifier [training model]
        * dumps EIMP score matrix [training model]

    2. screen [const PKL_FILE NOT NECESSARY]
        get PKL_FILE from const or from variable while pre-process running
        from PKL_FILE loads reader
        loads EIMP score matrix from const
        dumps screened data [screening option]
        * dumps screened eval results [analysis option]
    
    // ClassifyWorkFlow //
    
    3. extract 
        get PKL_FILE from const
        from PKL_FILE loads reader
        * dumps ensemble LDA classifier [training model / validating model]
        * loads ensemble LDA classifier [testing model]
        dumps extracted data

    4. classify
        get PKL_FILE from const
        from PKL_FILE load reader
        * dumps ensemble SVM classifier [training model / validating model]
        * dumps classifier eval results [training model / validating model]

        * loads ensemble SVM classifier [testing model]
        * dumps classified data [testing model]
    
    // ValidationWorkFlow //
    
    5. validate
        get PKL_FILE from const
        from PKL_FILE load reader
        
        reload reader through kernel algorithm
        validate eval workflow
"""


class FeaturesProjectWorkFlow(object):

    def __init__(self, file_name):
        self.file_name = file_name

        self.training = 1

        self.flush = False
        self.pklfile = None
        self.test_size = None
        self.eim_choice = list()

        self.reader = None

    def eim(self):
        eim = EMBTABinomalEIMProcess(self.file_name, pklfile=self.pklfile)
        # dataReader Flushed
        if self.flush:
            eim.flushed()

        eim.test_size = self.test_size
        # Calculate Equ_importance Integral Matrix
        eim.execute_eim_process(self.training)
        return self

    def eim_analysis(self):
        # Get Analysis Handler
        aly = EMBTABinomalEIMAnalysis(pklfile=self.pklfile)
        # Get Selected Expression Matrix and Significant Score Matrix
        # Get the Accuracy Matrix between Raw Expression and Selected Expression Matrix
        aly.execute_eim_analysis(*self.eim_choice, callback=self._callback)
        return self

    def _callback(self, reader):
        self.reader = reader


class FeaturesProjectPreWorkFlow(FeaturesProjectWorkFlow):

    def eim(self):
        eim = EMBTABinomalEIMProcess(self.file_name, pklfile=self.pklfile)
        # predict dataset use BinomialPredictDataReader
        eim.data_reader_class = BinomialPredictDataReader
        eim.rdparams = dict()

        # dataReader Flushed
        if self.flush:
            eim.flushed()

        eim.test_size = self.test_size
        # Calculate Equ_importance Integral Matrix
        eim.execute_eim_process(self.training)
        return self


class ClassifyWorkFlow(object):

    viewer2d = True

    def __init__(self, reader=None):
        self.reader = reader
        self.training = 1

    def estimate(self, **kwargs):
        if isinstance(self.reader, str):
            fext = EMBTABinomalDimensionEstimate(pklfile=self.reader)
        else:
            fext = EMBTABinomalDimensionEstimate()
            fext.reader = self.reader

        # n_components=12, after_filter=120, barnes_hut=0.5
        fext.execute_estimate_process(viewer2d=self.viewer2d,
                                      callback=self._callback,
                                      training=self.training,
                                      **kwargs)
        return self

    def classify(self, **kwargs):
        clf = EMBTASVMProcess()
        if self.reader:
            clf.reader = self.reader

        # n_estimator=132, record_freq=10
        clf.execute_classify_process(callback=self._callback,
                                     training=self.training,
                                     **kwargs)

    def _callback(self, reader):
        self.reader = reader


def workflow_dc(func):
    mapping = {
        1: FeaturesProjectWorkFlow,
        -1: FeaturesProjectWorkFlow,
        0: FeaturesProjectPreWorkFlow,
    }

    def inner(*args, **kwargs):
        mod = kwargs.pop('mod', 1)
        workflow_cls = mapping.get(mod, mapping[1])
        return func(workflow_cls, *args, **kwargs)

    return inner


@workflow_dc
def features(workflow_cls, filename=None, pklfile=None, flush=False, training=1, test_sz=0.2, aly_choice=()):
    # config filename
    if filename is None or not isinstance(filename, str):
        from embdata.data_const import RAW_DATA
        filename = RAW_DATA

    # init features screened workflow obj
    workflow = workflow_cls(filename)

    # config workflow params
    workflow.flush = flush
    workflow.pklfile = pklfile
    workflow.training = training
    workflow.test_size = test_sz
    choice = ["trans_and_sig", "acc_between_select"]
    aly_choice = aly_choice or choice
    if not set(aly_choice) & set(choice):
        print("Wrong params: %s" % aly_choice)
        return

    workflow.eim_choice = set(aly_choice) & set(choice)

    # start features screened workflow
    workflow.eim().eim_analysis()

    return workflow.reader


def classify(reader=None, training=1, viewer2d=False, **kwargs):
    # init features screened workflow obj
    workflow = ClassifyWorkFlow(reader)

    # config workflow params
    workflow.training = training
    workflow.viewer2d = viewer2d

    # resolute processor params
    fext_params = ['n_components', 'after_filter', 'barnes_hut']
    clf_params = ['n_estimator', 'record_freq', 'best_n']
    fextp = {k: kwargs[k] for k in fext_params if kwargs.get(k)}
    clfp = {k: kwargs[k] for k in clf_params if kwargs.get(k)}

    # start features screened workflow
    workflow.estimate(**fextp).classify(**clfp)

    return workflow.reader


def fitmodel(filename=None, pklfile=None, flush=False, trscreen=1, trclassify=1, test_sz=0.2, aly_choice=(), **kwargs):
    reader = features(filename, pklfile, flush, training=trscreen, test_sz=test_sz, aly_choice=aly_choice)
    reader = classify(reader, training=trclassify, **kwargs)
    return reader


class ValidationWorkFlow(object):

    viewer2d = False

    def __init__(self, reader):
        self.reader = reader

    def __call__(self, **kwargs):
        classify(self.reader, -1, self.viewer2d, **kwargs)


def validations(viewer2d=False, **kwargs):
    validator = EMBTABinomalValidator(pklfile=kwargs.pop('reader'))
    if kwargs.get('s'):
        validator.s = kwargs.pop('s')
    validator.workflow = ValidationWorkFlow
    validator.workflow.viewer2d = viewer2d
    # n_components=12, after_filter=120, barnes_hut=0.5, n_estimator=132, record_freq=10
    validator.validate(**kwargs)
