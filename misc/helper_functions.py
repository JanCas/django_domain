def update_model_scores(model, params, history, eval, metrics):
    from apps.training.models.ModelScores import ModelScores

    params['epochs'] = model.epochs
    params['batch_size'] = model.batch_size

    kwargs_train = {
        'model': model,
        'run': 1,
        'loss_score': history.history['loss'][-1],
        'MAE': history.history['mean_absolute_error'][-1],
        'MSE': history.history['mean_squared_error'][-1],
    }
    ModelScores.objects.get_or_create(**kwargs_train, **params)

    kwargs_val = {
        'model': model,
        'run': 2,
        'loss_score': history.history['val_loss'][-1],
        'MAE': history.history['val_mean_absolute_error'][-1],
        'MSE': history.history['val_mean_squared_error'][-1]
    }
    ModelScores.objects.get_or_create(**kwargs_val, **params)

    kwargs_test = {
        'model': model,
        'run': 0,
        'loss_score': eval[0],
        'MAE': eval[metrics.index('mean_absolute_error') + 1],
        'MSE': eval[metrics.index('mean_squared_error') + 1]
    }
    ModelScores.objects.get_or_create(**kwargs_test, **params)