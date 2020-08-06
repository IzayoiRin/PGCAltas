# from rest_framework.generics import GenericAPIView
from rest_framework.generics import GenericAPIView
from rest_framework.response import Response
from rest_framework.viewsets import GenericViewSet
from rest_framework.decorators import action

from embdata.serializers import FeaturesSerializer, ClassifySerializer, FittingSerializer, ValidatingSerializer, \
    PredcitSerializer

__DATA_ROOT__ = 'dataset\EMTAB6967'


# class FittingModelsAPIView(GenericAPIView):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         from embdata import patch_const
#         patch_const()
#         from embdata.workflow import features
#         self.features = features
#
#     # /embdata/fitting/?flush=1&screened=0&train=1&n_components=12&after_filter=120&barnes_hut=0.5&test_size=0.3
#     def get(self, request):
#         query_dict = request.query_params
#         # TODO: Serializer optim
#         query = {k: eval(v) for k, v in query_dict.items()}
#         self.features(**query)
#         # eim_mtx_file = os.path.join(__DATA_ROOT__, 'texts', 'RDFBinomialFlow.txt')
#         # with open(eim_mtx_file, 'r', encoding='utf-8') as f:
#         #     text = f.read()
#         # return HttpResponse(text, content_type="text/plain")
#         return Response({"msg": "Finish"})


class FittingModelsAPIViewSet(GenericViewSet):

    serializer_class = {
        "features": FeaturesSerializer,
        "classify": ClassifySerializer,
        "fitting": FittingSerializer,
        "predict": PredcitSerializer,
    }

    lookup_url_kwarg = 'mod'

    def __init__(self, *args, **kwargs):
        super(FittingModelsAPIViewSet, self).__init__(*args, **kwargs)
        from embdata import patch_const
        patch_const()
        from embdata.workflow import features as f
        from embdata.workflow import classify as c
        from embdata.workflow import fitmodel as m
        self.fprocess = f
        self.cprocess = c
        self.mprocess = m

    def get_serializer_class(self):
        # get url from request
        # url = self.request.path_info  # type: str
        mod = self.kwargs[self.lookup_url_kwarg]
        return self.serializer_class[mod]

    # /embdata/fitting/predict/?flush=1&best_n=0
    @action(methods=['get', 'post'], detail=False)
    def predict(self, request, **kwargs):
        query_dict = request.query_params
        if request.data:
            query_dict = query_dict.copy()
            for k, v in request.data.items():
                query_dict[k] = v

        s = self.get_serializer(data=query_dict)
        reader = s.calling(self.fprocess, self.cprocess)
        return Response({"msg": "Finish: %s" % reader.pklname})

    # /embdata/fitting/features/?flush=1&training=1&test_sz=0.2
    @action(methods=['get', 'post'], detail=False)
    def features(self, request, **kwargs):
        query_dict = request.query_params

        if request.data:
            query_dict = query_dict.copy()
            for k, v in request.data.items():
                query_dict[k] = v

        s = self.get_serializer(data=query_dict)
        reader = s.calling(self.fprocess)
        return Response({"msg": "Finish: %s" % reader.pklname})

    # /embdata/fitting/classify/?training=1&n_components=12&after_filter=120&barnes_hut=0.5&n_estimator=132&record_freq=10
    @action(methods=['post'], detail=False)
    def classify(self, request, **kwargs):
        query_dict = request.query_params

        if request.data:
            query_dict = query_dict.copy()
            for k, v in request.data.items():
                query_dict[k] = v

        s = self.get_serializer(data=query_dict)
        reader = s.calling(self.cprocess)
        return Response({"msg": "Finish: %s" % reader.pklname})

    # /embdata/fitting/?flush=1&trscreen=1&trclassify=1&test_sz=0.2&n_components=12&after_filter=120&barnes_hut=0.5&n_estimator=132&record_freq=10
    def create(self, request, **kwargs):
        query_dict = request.query_params

        if request.data:
            query_dict = query_dict.copy()
            for k, v in request.data.items():
                query_dict[k] = v

        s = self.get_serializer(data=query_dict)
        reader = s.calling(self.mprocess)
        return Response({"msg": "Finish: %s" % reader.pklname})


class ValidationsAPIView(GenericAPIView):

    serializer_class = ValidatingSerializer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from embdata import patch_const
        patch_const()
        from embdata.workflow import validations as v
        self.vprocess = v

    # /embdata/validations/?n_components=12&after_filter=120&barnes_hut=0.5&n_estimator=132&record_freq=10
    def post(self, request):
        query_dict = request.query_params

        if request.data:
            query_dict = query_dict.copy()
            for k, v in request.data.items():
                query_dict[k] = v

        query_dict['training'] = -1
        s = self.get_serializer(data=query_dict)
        s.calling(self.vprocess)
        return Response({"msg": "Finish"})
