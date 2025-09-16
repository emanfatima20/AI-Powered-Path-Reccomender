from django.urls import path
from . import views

urlpatterns = [
    path("", views.recommender_view, name="chatbot"),  # root URL → chatbot
]
