from django.db import models


class ModelScores(models.Model):
    RUN_CHOICES = [
        (0, 'test'),
        (1, 'train'),
        (2, 'val')
    ]

    model = models.ForeignKey('training.Model', on_delete=models.CASCADE)
    run = models.SmallIntegerField(default=0, choices=RUN_CHOICES, null=False)
    loss = models.FloatField(default=None, null=True)
    MAE = models.FloatField(default=None, null=True)
    MSE = models.FloatField(default=None, null=True)
    R2 = models.FloatField(default=None, null=True)

    class Meta:
        verbose_name_plural = 'Model Scores'

    def __str__(self):
        return self.RUN_CHOICES[self.run][-1]
