from rest_framework.views import exception_handler as drf_exhandler
import logging
from django.db import DatabaseError
from redis.exceptions import RedisError
from rest_framework.response import Response
from rest_framework import status


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
            logger.error("[%s]: %s" % (view, exc))
            response = Response({"msg": 'INTERNAL SERVER ERROR'}, status=status.HTTP_507_INSUFFICIENT_STORAGE)
    return response
