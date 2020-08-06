import os
import django
from django.conf import LazySettings

if os.environ.get("DJANGO_SETTINGS_MODULE", None) is None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "PGCAltas.settings.dev")
    django.setup()

settings = LazySettings()
PATH_ = os.path.dirname(settings.BASE_DIR) + settings.DATASET_URL
TORCH_PATH_ = os.path.join(PATH_, settings.TH_DATASET)
