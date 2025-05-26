from app02 import views
from django.urls import path

urlpatterns = [
    path('index/', views.index, name='index'),
    path('users_list/', views.users_list, name='users_list'),
    path('orm/', views.orm, name='orm'),
    path('info_list/', views.info_list, name='info_list'),
    path('login/', views.login, name='login'),
    path('info_delete/', views.info_delete, name='info_delete'),
    path('model_pf/', views.model_pf, name='model_pf'),
    path('upload_excel/', views.upload_excel, name='upload_excel'),
    path('upload_excel2/', views.upload_excel2, name='upload_excel2'),
    path('show_image/', views.show_image, name='show_image'),
    path('use_model/<int:model_id>/', views.use_model, name='use_model')
    ]