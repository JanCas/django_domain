def create_models():
    from apps.training.models.Model import Model
    from apps.training.models.ModelParams import ModelParams
    from apps.training.models.Metrics import Metrics
    from apps.training.models.Callbacks import Callbacks

    for cbfv in Model.CBFV_CHOICES:
        for mat_prop in Model.MAT_PROP_CHOICES:

            print('------------------')
            model_kwargs = {
                'name': 'untrained',
                'cbfv': cbfv[0],
                'mat_prop': mat_prop[0]
            }
            model = Model.objects.get_or_create(**model_kwargs)[0]
            print('Created model with {} and {}'.format(cbfv[0], mat_prop[1]))

            ModelParams.objects.get_or_create(**{'model':model})
            print('creating model params')

            Metrics.objects.get_or_create(**{'model':model})
            Metrics.objects.get_or_create(**{'model': model, 'metric':2})
            print('creating metrics')

            Callbacks.objects.get_or_create(**{'model':model})
            Callbacks.objects.get_or_create(**{'model':model, 'name':1, 'patience':15})
            print('creating callbacks')
            print()