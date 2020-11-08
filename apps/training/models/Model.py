from django.db import models


class Model(models.Model):
    CBFV_CHOICES = [
        (0, 'atom2vec'),
        (1, 'jarvis'),
        (2, 'jarvis_shuffled'),
        (3, 'magpie'),
        (4, 'mat2vec'),
        (5, 'mat_shuffled'),
        (6, 'oliynyk'),
        (7, 'random_200')
    ]

    MAT_PROP_CHOICES = [
        (0, 'Egap'),
        (1, 'ael_bulk_modulus_vrh'),
        (2, 'ael_debye_temperature'),
        (3, 'ael_shear_modulus_vrh'),
        (4, 'agl_log10_termal_expansion_300K'),
        (5, 'agl_thermal_conductivity_300K'),
        (6, 'energy_atom')
    ]

    name = models.CharField(default=None, max_length=100, null=True)
    cbfv = models.SmallIntegerField(default=0, choices=CBFV_CHOICES, null=False)
    mat_prop = models.SmallIntegerField(default=0, choices=MAT_PROP_CHOICES, verbose_name='material property',
                                        null=False)
    batch_size = models.IntegerField(default=64, null=False)
    epochs = models.IntegerField(default=200, null=False)
    model_graph = models.ImageField

    learning_graph = models.ImageField(upload_to='graphs', null=True)

    def __str__(self):
        return self.name

    def train(self):
        from .ModelParams import ModelParams
        from .Metrics import Metrics
        from .Callbacks import Callbacks
        from .ModelScores import ModelScores

        from misc.conv_net import Alex_Net
        from misc.pandas_creator import generate_image_data_generators

        from importlib import import_module
        from numpy import floor

        from matplotlib.pyplot import subplots

        from tensorflow.compat.v1 import ConfigProto, Session

        # setting the name
        self.set_name()

        # settign the GPU
        config = ConfigProto
        config.gpu_options.per_process_gpu_memory_fraction = 1
        session = Session(config=config)

        # getting the data
        data = generate_image_data_generators(material_prop=self.mat_prop, cbfv=self.cbfv, batch_size=self.batch_size)

        model_params = ModelParams.objects.get(model=self).get_dict()

        metrics = []
        for metric in Metrics.objects.filter(model=self):
            metrics.append(metric.get_metric())

        callbacks = []
        for callback in Callbacks.objects.filter(model=self):
            callbacks.append(getattr(import_module('.callbacks', 'tensorflow.keras'), str(callback))(monitor=callback.monitor, patience=callback.patience))

        opt = getattr(import_module('.optimizers', 'tensorflow.keras'), model_params['optimizer'])(
            model_params['learning_rate'])

        model = Alex_Net(input=data['input'], regularization=self.REGULARIZATION_CHOICES[self.regularization][-1],
                         dropout=self.dropout)

        model.compile(loss=model_params['loss'], optimizer=opt, metrics=metrics)

        history = model.fit(data['train'],
                            epochs=self.epochs,
                            steps_per_epoch=floor(data['train'].n / data['train'].batch_size),
                            validation_data=data['val'],
                            callbacks=callbacks)

        # saving the training image
        fig, ax = subplots(3, 1)
        ax[0].plot(history.history['loss'][3:], color='b', label="Training loss")
        ax[0].plot(history.history['val_loss'][3:], color='r', label="validation loss", axes=ax[0])
        ax[0].legend(loc='best', shadow=True)

        ax[1].plot(history.history['mean_squared_error'][3:], color='b', label="mean squared error")
        ax[1].plot(history.history['val_mean_squared_error'][3:], color='r', label="Validation MSE")
        ax[1].legend(loc='best', shadow=True)

        ax[2].plot(history.history['mean_absolute_error'][3:], color='b', label='mean absolute error')
        ax[2].plot(history.history['val_mean_absolute_error'][3:], color='r', label='val MAE')
        ax[2].legend(loc='best', shadow=True)

        #saving the training data
        kwargs_train = {
            'model': self,
            'run': 1,
            'loss': history.history['loss'][-1],
            'MAE': history.history['mean_absolute_error'][-1],
            'MSE': history.history['mean_squared_error'][-1]
        }
        ModelScores.objects.get_or_create(**kwargs_train)

        kwargs_val = {
            'model': self,
            'run': 2,
            'loss': history.history['val_loss'][-1],
            'MAE': history.history['val_mean_absolute_error'][-1],
            'MSE': history.history['val_mean_squared_error'][-1]
        }
        ModelScores.objects.get_or_create(**kwargs_val)

        

    def set_name(self):
        self.name = '{} -> {}'.format(self.cbfv, self.mat_prop)
        self.save()
