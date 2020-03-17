# DataReader's Error
class OSPathError(Exception):
    pass


class DataFrameFitError(Exception):
    pass


class ReaderLoadError(Exception):
    pass


# Processor's Error
class MessProcessesError(Exception):
    pass


class FailInitialedError(Exception):
    pass


# Classifier's Error
class ModelProcessError(Exception):
    pass


class CannotMoveError(Exception):
    pass


class CannotAnalysisError(Exception):
    pass


# Validator's Error
class WorkFlowInitiationError(Exception):
    pass


class WorkFlowRunningError(Exception):
    pass
