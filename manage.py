#!/usr/bin/env python
import os
import sys


def fire_resoluter(cmapping):
    cmd_args = sys.argv
    startf = cmd_args[0]
    cmd = cmapping[cmd_args[1]]
    module, calling = cmd.rsplit('.', 1)
    from importlib import import_module
    calling = getattr(import_module(module), calling)
    import fire
    fire.Fire({cmd_args[1]: calling})


if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "PGCAltas.settings.dev")
    try:
        from django.core.management import execute_from_command_line
    except ImportError:
        # The above import may fail for some other reason. Ensure that the
        # issue is really that Django is missing to avoid masking other
        # exceptions on Python 2.
        try:
            import django
        except ImportError:
            raise ImportError(
                "Couldn't import Django. Are you sure it's installed and "
                "available on your PYTHONPATH environment variable? Did you "
                "forget to activate a virtual environment?"
            )
        raise
    from django.conf import LazySettings

    settings = LazySettings()
    if len(sys.argv) > 1 and sys.argv[1] in settings.COMMANDS:
        import django
        django.setup()
        fire_resoluter(settings.COMMANDS)
    else:
        execute_from_command_line(sys.argv)
