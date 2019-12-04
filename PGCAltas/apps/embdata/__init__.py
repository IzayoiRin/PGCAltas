def patch_const():
    from importlib import reload, import_module

    const = import_module("PGCAltas.utils.StatExpr.const")
    setattr(const, 'module', 'embdata.data_const')
    reload(const)
    print('Patched from: embdata.data_const')
