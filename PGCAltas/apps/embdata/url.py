from django.conf.urls import url

from . import views


urlpatterns = [
    url(r"^features/screen/$", views.FeaturesScreenAPIView.as_view())
]
