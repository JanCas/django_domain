from django.db import models

class Callbacks(models.Model):
    NAME_CHOICES = [
        (0, 'ReduceLROnPlateau'),
        (1, 'EarlyStopping')
    ]

    model = models.ForeignKey('training.Model', on_delete=models.CASCADE)
    name = models.SmallIntegerField(default=0, choices=NAME_CHOICES ,null=False)
    monitor = models.CharField(default='val_loss', null=False, max_length=50)
    patience = models.IntegerField(default=3, null=False)

    class Meta:
        verbose_name_plural = 'Callbacks'

    def __str__(self):
        return self.NAME_CHOICES[self.name][-1]