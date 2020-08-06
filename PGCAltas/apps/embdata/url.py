from django.conf.urls import url

from . import views


# urlpatterns = [
#     url(r"^fitting/$", views.FittingModelsAPIView.as_view()),
#     url(r"^validations/$", views.ValidationsAPIView.as_view()),
# ]
urlpatterns = [
    url(r"^fitting/(?P<mod>features)/$",
        views.FittingModelsAPIViewSet.as_view({'get': 'features', 'post': 'features'}), name='fit-features'),
    url(r"^fitting/(?P<mod>classify)/$",
        views.FittingModelsAPIViewSet.as_view({'post': 'classify'}), name='fit-classify'),
    url(r"^(?P<mod>fitting)/$", views.FittingModelsAPIViewSet.as_view({'post': 'create'}), name='fit-model'),
    url(r"^fitting/(?P<mod>predict)/$",
        views.FittingModelsAPIViewSet.as_view({'get': 'predict', 'post': 'predict'}), name='fit-predict'),
    url(r"^validations/$", views.ValidationsAPIView.as_view(), name='validation'),
]


"""
test cases:

1. fitting features -- EMIP
    
    Only for NEW Training data
    /embdata/fitting/features/?flush=1&training=1&test_sz=0.2
    
    Only for NEW Testing data
    /embdata/fitting/features/?flush=1&training=0&test_sz=0.2
    
    Used for redefined RDF Selector from EXISTED data
    /embdata/fitting/features/?flush=0&training=1&test_sz=0.2
    
    A way passing this stage
    /embdata/fitting/features/?flush=0&training=0&test_sz=0.2

2. fitting classifier -- ENSEMBLE CLASSIFIER

    Used for redefined ENSEMBLE CLASSIFIER from EXISTED data
    /embdata/fitting/classify/?training=1&n_components=12&after_filter=120&barnes_hut=0.5&n_estimator=132&record_freq=10
    
    Only for PREDICT form EXISTED data and fitted model
    /embdata/fitting/classify/?training=0&n_components=12&after_filter=120&barnes_hut=0.5&n_estimator=132&record_freq=10
    
    Used for redefined ENSEMBLE CLASSIFIER from EXISTED data, CHANGE PART: LDA
    /embdata/fitting/classify/?training=1&n_components=12&after_filter=90&barnes_hut=0.5&n_estimator=132&record_freq=10
    
    Used for redefined ENSEMBLE CLASSIFIER from EXISTED data, CHANGE PART: SVM
    /embdata/fitting/classify/?training=1&n_components=12&after_filter=90&barnes_hut=0.5&n_estimator=87&record_freq=10
    
3. fitting -- WHOLE WORKFLOW

    Only for NEW Training data, Used for redefined ENSEMBLE CLASSIFIER from EXISTED data
    /embdata/fitting/?flush=1&trscreen=1&trclassify=1&test_sz=0.2&n_components=12&after_filter=120&barnes_hut=0.5&n_estimator=132&record_freq=10
    
    # Only for NEW Training data, PREDICT form EXISTED data and fitted model
    # /embdata/fitting/?flush=1&trscreen=1&trclassify=0&test_sz=0.2&n_components=12&after_filter=120&barnes_hut=0.5&n_estimator=132&record_freq=10
        
    Only for Screened NEW Training data, Used for redefined ENSEMBLE CLASSIFIER from EXISTED data
    /embdata/fitting/?flush=1&trscreen=0&trclassify=1&test_sz=0.2&n_components=12&after_filter=120&barnes_hut=0.5&n_estimator=132&record_freq=10
    
    Only for NEW Testing data, PREDICT form EXISTED data and fitted model
    /embdata/fitting/?flush=1&trscreen=0&trclassify=0&test_sz=0.2&n_components=12&after_filter=120&barnes_hut=0.5&n_estimator=132&record_freq=10
    
    Used for redefined RDF Selector from EXISTED data, and redefined ENSEMBLE CLASSIFIER from EXISTED data
    /embdata/fitting/?flush=0&trscreen=1&trclassify=1&test_sz=0.2&n_components=12&after_filter=120&barnes_hut=0.5&n_estimator=132&record_freq=10
    
    # Used for redefined RDF Selector from EXISTED data, and PREDICT form EXISTED data and fitted model
    # /embdata/fitting/?flush=0&trscreen=1&trclassify=0&test_sz=0.2&n_components=12&after_filter=120&barnes_hut=0.5&n_estimator=132&record_freq=10
    
    A way passing this stage, Used for redefined ENSEMBLE CLASSIFIER from EXISTED data
    /embdata/fitting/?flush=0&trscreen=0&trclassify=1&test_sz=0.2&n_components=12&after_filter=120&barnes_hut=0.5&n_estimator=132&record_freq=10
    
    A way passing this stage, PREDICT form EXISTED data and fitted model
    /embdata/fitting/?flush=0&trscreen=0&trclassify=0&test_sz=0.2&n_components=12&after_filter=120&barnes_hut=0.5&n_estimator=132&record_freq=10

4. validating -- S-FOLD VALIDATION
    
    VALIDATING 
    /embdata/validations/?n_components=12&after_filter=120&barnes_hut=0.5&n_estimator=132&record_freq=10
    
    VALIDATING, CHANGE PART: SVM
    /embdata/validations/?n_components=12&after_filter=120&barnes_hut=0.5&n_estimator=98&record_freq=10
    
    VALIDATING, CHANGE PART: LDA
    /embdata/validations/?n_components=12&after_filter=80&barnes_hut=0.5&n_estimator=98&record_freq=10
"""



