def update_model_scores(model, history, eval, metrics):
    from apps.training.models.ModelScores import ModelScores

    kwargs_train = {
        'model': model,
        'run': 1,
        'loss': history.history['loss'][-1],
        'MAE': history.history['mean_absolute_error'][-1],
        'MSE': history.history['mean_squared_error'][-1]
    }
    ModelScores.objects.get_or_create(**kwargs_train)

    kwargs_val = {
        'model': model,
        'run': 2,
        'loss': history.history['val_loss'][-1],
        'MAE': history.history['val_mean_absolute_error'][-1],
        'MSE': history.history['val_mean_squared_error'][-1]
    }
    ModelScores.objects.get_or_create(**kwargs_val)

    kwargs_test = {
        'model': model,
        'run': 0,
        'loss': eval[0],
        'MAE': eval[metrics.index('mean_absolute_error') + 1],
        'MSE': eval[metrics.index('mean_squared_error') + 1]
    }
    ModelScores.objects.get_or_create(**kwargs_test)