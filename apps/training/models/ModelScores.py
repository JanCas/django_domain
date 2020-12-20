from django.db import models


class ModelScores(models.Model):
    RUN_CHOICES = [
        (0, 'test'),
        (1, 'train'),
        (2, 'val')
    ]

    model = models.ForeignKey('training.Model', on_delete=models.CASCADE)
    run = models.SmallIntegerField(default=0, choices=RUN_CHOICES, null=False)
    loss_score = models.FloatField(default=None)
    MAE = models.FloatField(default=None)
    MSE = models.FloatField(default=None)
    optimizer = models.CharField(default=None, max_length=25)
    loss = models.CharField(default=None, max_length=25)
    learning_rate = models.FloatField(default=.01)
    regularization = models.CharField(default=None, max_length=25)
    dropout = models.FloatField(default=None)
    augmentation = models.BooleanField(default=False)
    epochs = models.IntegerField(default=200)
    batch_size = models.IntegerField(default=25)
    model_type = models.CharField(default=None, max_length=25, null=True)


    class Meta:
        verbose_name_plural = 'Model Scores'

    def __str__(self):
        return self.RUN_CHOICES[self.run][-1]
