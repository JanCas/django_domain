from django.db import models

class Metrics(models.Model):
    METRICS_CHOICES = [
        (0, 'mean_squared_error'),
        (1, 'root_mean_squared_error'),
        (2, 'mean_absolute_error'),
        (3, 'mean_absolute_percentage_error'),
        (4, 'mean_squared_logarithmic_error'),
        (5, 'cosine_similarity'),
        (6, 'logcosh')
    ]

    model = models.ForeignKey('training.Model', on_delete=models.CASCADE)
    metric = models.SmallIntegerField(default=1, choices=METRICS_CHOICES)

    class Meta:
        verbose_name_plural = 'Metrics'


    def get_metric(self):
        return self.METRICS_CHOICES[self.metric][-1]