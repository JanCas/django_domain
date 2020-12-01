from django.db import models


class ModelParams(models.Model):
    OPTIMIZER_CHOICES = [
        (0, 'Adam'),
        (1, 'SGD')
    ]

    LOSS_CHOICES = [
        (0, 'huber_loss'),
        (1, 'mean_absolute_error'),
        (2, 'mean_squared_error')
    ]

    REGULARIZATION_CHOICES = [
        (0, None),
        (1, 'l2'),
        (2, 'l1')
    ]

    MODEL_TYPE_CHOICES = [
        (0, 'alex_net'),
        (1, 'light_alex_net')
    ]

    model = models.ForeignKey('training.Model', on_delete=models.CASCADE)
    model_type = models.SmallIntegerField(default=0, choices=MODEL_TYPE_CHOICES)
    optimizer = models.SmallIntegerField(default=0, choices=OPTIMIZER_CHOICES)
    loss = models.SmallIntegerField(default=0, choices=LOSS_CHOICES)
    learning_rate = models.FloatField(default=.01)
    regularization = models.SmallIntegerField(default=1, choices=REGULARIZATION_CHOICES)
    dropout = models.FloatField(default=.4)

    class Meta:
        verbose_name_plural = 'Model Params'

    
    def get_dict(self):
        return {'optimizer': self.OPTIMIZER_CHOICES[self.optimizer][-1],
                'loss': self.LOSS_CHOICES[self.loss][-1],
                'learning_rate': self.learning_rate,
                'regularization': self.REGULARIZATION_CHOICES[self.regularization][-1],
                'dropout': self.dropout,
                'model_type': self.MODEL_TYPE_CHOICES[self.model_type][-1]}

    def set_model_type(self, choice):
        self.model_type = choice
        self.save()