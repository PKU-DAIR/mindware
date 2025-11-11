from django.db import models


# Create your models here.
class Task(models.Model):
    task_id = models.CharField(max_length=50)
    task_type = models.CharField(max_length=50)
    opt_type = models.CharField(max_length=50)
    task_start_time = models.DateTimeField(primary_key=True)
    task_end_time = models.DateTimeField()

    task_info = models.TextField()

    def __str__(self):
        return self.task_id


class Observation(models.Model):
    task_id = models.CharField(max_length=50)

    idx = models.IntegerField()
    configuration = models.TextField()
    performance = models.FloatField()
    time_consumed = models.FloatField()

    def __str__(self):
        return self.task_id
