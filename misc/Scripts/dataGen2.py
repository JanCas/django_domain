def create_results_csv():
    from pandas import DataFrame
    from os.path import join

    from django_domain.settings import BASE_DIR
    from apps.training.models.Model import Model
    from apps.training.models.ModelScores import ModelScores

    result_dict = {
        'model': [],
        'run': [],
        'loss_score': [],
        'MAE': [],
        'MSE': [],
        'optimizer': [],
        'loss': [],
        'learning_rate': [],
        'regularization': [],
        'dropout': [],
        'epochs': [],
        'batch_size': [],
    }
    for model in Model.objects.all():
        for model_score in ModelScores.objects.filter(model=model):
            result_dict['model'].append(str(model))
            for k, v in result_dict.items():
                if k == 'run':
                    result_dict[k].append(str(model_score))
                elif k is not 'model':
                    result_dict[k].append(getattr(model_score, k))

    df = DataFrame(result_dict)
    result_dir = join(BASE_DIR, 'results', 'results.csv')
    df.to_csv(result_dir)