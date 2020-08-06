from urllib import parse
from collections import OrderedDict

from rest_framework import serializers


class BaseAbstractSerializer(serializers.Serializer):
    # training = 1
    training = serializers.IntegerField(required=True)

    ALL_HOLDPLACE = '__all__'

    def get_fields(self):
        ret = super(BaseAbstractSerializer, self).get_fields()  # type: OrderedDict

        if not hasattr(self, 'Meta'):
            return ret

        if hasattr(self.Meta, 'fields') and self.Meta.fields != self.ALL_HOLDPLACE:
            ret = OrderedDict([(k, ret[k]) for k in self.Meta.fields])

        if hasattr(self.Meta, 'exclude'):
            for i in self.Meta.exclude:
                if ret.get(i):
                    ret.pop(i)
        return ret

    def calling(self, fn):
        if self.is_valid(raise_exception=True):
            return fn(**self.validated_data)


class FeaturesSerializer(BaseAbstractSerializer):
    # flush = 1 & test_sz = 0.
    filename = serializers.CharField(required=False)
    flush = serializers.IntegerField(required=True)
    test_sz = serializers.FloatField(required=False)

    def validate(self, attrs):
        if attrs.get('filename') and not attrs['flush']:
            attrs['pklfile'] = attrs.pop('filename')
        return attrs


class ClassifySerializer(BaseAbstractSerializer):
    reader = serializers.CharField(required=True)

    # n_components = 12 & after_filter = 120 & barnes_hut = 0.5
    n_components = serializers.IntegerField(required=True)
    after_filter = serializers.IntegerField(required=True)
    barnes_nut = serializers.FloatField(required=False)

    # n_estimator = 132 & record_freq = 10
    n_estimator = serializers.IntegerField(required=True)
    record_freq = serializers.IntegerField(required=False)
    best_n = serializers.IntegerField(required=False)


class FittingSerializer(FeaturesSerializer, ClassifySerializer):
    # trscreen = 1 & trclassify = 1
    trscreen = serializers.IntegerField(required=True)
    trclassify = serializers.IntegerField(required=True)

    class Meta:

        exclude = ['training', 'reader']

    def validate(self, attrs):
        trscreen, trclassify = attrs['trscreen'], attrs['trclassify']
        if trscreen > trclassify:
            raise serializers.ValidationError("Wrong workflow: define Selector then predict by EXISTED Classifier")
        if attrs.get('filename') and not attrs['flush']:
            attrs['pklfile'] = attrs.pop('filename')
        return attrs


class PredcitSerializer(BaseAbstractSerializer):
    # flush = 1 & test_sz = 0.
    filename = serializers.CharField(required=False)
    flush = serializers.IntegerField(required=True)

    n_components = serializers.IntegerField(required=True)
    after_filter = serializers.IntegerField(required=True)
    best_n = serializers.IntegerField(required=False)

    class Meta:

        exclude = ['training']

    def validate(self, attrs):
        if attrs.get('filename') and not attrs['flush']:
            attrs['pklfile'] = attrs.pop('filename')
        return attrs

    def calling(self, *fn):
        f, c = fn
        if self.is_valid(raise_exception=True):
            fp = {k: self.validated_data.pop(k) for k in ['filename', 'flush', 'pklfile'] if self.validated_data.get(k)}
            reader = f(mod=0, training=0, aly_choice=("trans_and_sig",), **fp)
            return c(reader, training=0, **self.validated_data)


class ValidatingSerializer(ClassifySerializer):
    reader = serializers.ListField(required=True)
    s = serializers.IntegerField(required=False)

    class Meta:
        exclude = ['training', 'best_n']

    def validate_reader(self, value):
        for i in value[0]:
            if not isinstance(i, str):
                raise serializers.ValidationError("Wrong Params: Reader Must be name")
        return value[0]
