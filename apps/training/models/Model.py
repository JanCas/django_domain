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
        (7, 'random_200'),
        (8, 'random_400')
    ]

    MAT_PROP_CHOICES = [
        (0, 'Egap'),
        (1, 'ael_bulk_modulus_vrh'),
        (2, 'ael_debye_temperature'),
        (3, 'ael_shear_modulus_vrh'),
        (4, 'agl_log10_thermal_expansion_300K'),
        (5, 'agl_thermal_conductivity_300K'),
        (6, 'energy_atom')
    ]

    name = models.CharField(default=None, max_length=100)
    cbfv = models.SmallIntegerField(default=0, choices=CBFV_CHOICES, null=False)
    mat_prop = models.SmallIntegerField(default=0, choices=MAT_PROP_CHOICES, verbose_name='material property',
                                        null=False)
    batch_size = models.IntegerField(default=64, null=False)
    epochs = models.IntegerField(default=200, null=False)

    trained = models.BooleanField(default=False, null=False)
    tf_error = models.BooleanField(default=False, null=False)
    memory_error = models.BooleanField(default=False, null=False)

    # model_graph = models.ImageField(null=True)

    # learning_graph = models.ImageField(upload_to='graphs', null=True)

    def __str__(self):
        return self.name

    def train(self):
        from .ModelParams import ModelParams
        from .Metrics import Metrics
        from .Callbacks import Callbacks

        from misc.helper_functions import update_model_scores
        from misc.pandas_creator import generate_image_data_generators

        from importlib import import_module
        from numpy import floor
        from gc import collect

        from tensorflow.compat.v1 import ConfigProto, Session
        import tensorflow as tf

        # setting the name
        self.set_name()
        self.trained = False
        self.save()

        # settign the GPU
        config = ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        session = Session(config=config)

        # clearing the session
        tf.keras.backend.clear_session()
        try:
            model_params = ModelParams.objects.get_or_create(model=self)[0].get_dict()

            # getting the data
            data = generate_image_data_generators(material_prop=self.MAT_PROP_CHOICES[self.mat_prop][-1],
                                                  cbfv=self.CBFV_CHOICES[self.cbfv][-1], batch_size=self.batch_size,
                                                  augmentation=model_params['augmentation'])

            metrics = []
            for metric in Metrics.objects.filter(model=self):
                metrics.append(metric.get_metric())

            callbacks = []
            for callback in Callbacks.objects.filter(model=self):
                callbacks.append(
                    getattr(import_module('.callbacks', 'tensorflow.keras'), str(callback))(monitor=callback.monitor,
                                                                                            patience=callback.patience))

            opt = getattr(import_module('.optimizers', 'tensorflow.keras'), model_params['optimizer'])(
                model_params['learning_rate'])

            model = getattr(import_module('misc.conv_net'), model_params['model_type'])(input=data['input'], regularization=model_params['regularization'],
                             dropout=model_params['dropout'])

            model.compile(loss=model_params['loss'], optimizer=opt, metrics=metrics)
            model.summary()

            history = model.fit(data['train'],
                                epochs=self.epochs,
                                steps_per_epoch=floor(data['train'].n / data['train'].batch_size),
                                validation_data=data['val'],
                                callbacks=callbacks)


            '''# saving the training image
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
            '''

            # evaluating the model
            eval = model.evaluate(data['test'])

            # saving the training data
            update_model_scores(self, model_params, history, eval, metrics)

            self.trained = True
            self.tf_error = False
            self.memory_error = False
            self.save()

            del model
            del data
        except tf.errors.ResourceExhaustedError:
            print('TF MEMORY EXHAUSTED ERROR')
            if self.batch_size > 4:
                self.batch_size /= 2
                self.tf_error = True
                self.save()
        except MemoryError:
            print('MEMORY ERROR')
            self.memory_error = True
            self.save()
        except ValueError as value_error:
            print('VALUE ERROR')
            print(value_error)
            ModelParams.objects.get_or_create(model=self)[0].set_model_type(1)
            self.tf_error = True
            self.save()

        collect()
        collect()

    def set_name(self):
        self.name = '{} -> {}'.format(self.CBFV_CHOICES[self.cbfv][-1], self.MAT_PROP_CHOICES[self.mat_prop][-1])
        self.save()