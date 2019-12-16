from rest_framework.generics import GenericAPIView
from rest_framework.response import Response

__DATA_ROOT__ = 'dataset\GSE120963'


class FeaturesScreenAPIView(GenericAPIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from gsedata import patch_const
        patch_const()
        from gsedata.workflow import features
        global features

    # embdata/features/?flush=1
    def get(self, request):
        query_dict = request.query_params
        # TODO: Serializer optim
        query = {k: eval(v) for k, v in query_dict.items()}
        features(**query)
        # eim_mtx_file = os.path.join(__DATA_ROOT__, 'texts', 'RDFBinomialFlow.txt')
        # with open(eim_mtx_file, 'r', encoding='utf-8') as f:
        #     text = f.read()
        # return HttpResponse(text, content_type="text/plain")
        return Response({"msg": "Finish"})

