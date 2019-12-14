from django.conf.urls import url

from . import views


urlpatterns = [
    url(r"^features/$", views.FeaturesScreenAPIView.as_view())
]
