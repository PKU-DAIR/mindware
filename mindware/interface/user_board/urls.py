# License: MIT

from django.urls import path

from . import views

app_name = 'user_board'
urlpatterns = [
    # /user_board/
    path('', views.index, name='index'),
    path('index/', views.index, name='index'),

    path('history/', views.history, name='history'),
    path('new_task/', views.new_task, name='new_task'),

    path('get_tasks/', views.get_tasks, name='get_tasks'),
    path('get_task_details/', views.get_task_details, name='get_task_details'),

    path('submit_task/', views.submit_task, name='submit_task'),
    path('show_process/', views.show_process, name='show_process'),

    path('delete_task/', views.delete_task, name='delete_task'),

]
