from rest_framework.views import exception_handler as drf_exhandler
import logging
from django.db import DatabaseError
from redis.exceptions import RedisError
from rest_framework.response import Response
from rest_framework import status

from .errors import *


logger = logging.getLogger('django')


def exception_handler(exc, context):
    """
    handle the exception from db_error and drf_error
    :param exc:
    :param context:
    :return:
    """
    response = drf_exhandler(exc, context)
    if response is None:
        view = context['view']
        if isinstance(exc, (DatabaseError, RedisError)):
            response = Response({"msg": 'INTERNAL SERVER ERROR'}, status=status.HTTP_507_INSUFFICIENT_STORAGE)
        if isinstance(exc, (MessProcessesError, ModelProcessError, FailInitialedError)):
            response = Response({"msg": 'INTERNAL SERVER ERROR'}, status=status.HTTP_400_BAD_REQUEST)
        if isinstance(exc, (CannotMoveError, CannotAnalysisError)):
            response = Response({"msg": "Model Fitting Failed: %s" % exc}, status=status.HTTP_400_BAD_REQUEST)
        logger.error("[%s]: %s" % (view, exc))
    return response
