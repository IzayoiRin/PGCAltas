if 'module' in dir():
    from importlib import import_module
    package = import_module(module)
