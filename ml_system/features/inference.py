import os


def shift_data(train_data):
    # sort values based on date and place
    train_data = train_data.sort_values(by=["date", 'place'], ascending=[False, True]).reset_index(drop=True)
    train_data["streamflow_next_day"] = train_data.groupby('place')['streamflow'].shift(1)
    train_data = train_data.drop(columns=["date"]).fillna(0)
    return train_data


def get_model(project, model_name, evaluation_metric, sort_metrics_by):
    import joblib
    """Retrieve desired model or download it from the Hopsworks Model Registry.

    In second case, it will be physically downloaded to this directory"""
    TARGET_FILE = "model.pkl"
    list_of_files = [os.path.join(dirpath, filename) for dirpath, _, filenames \
                     in os.walk('') for filename in filenames if filename == TARGET_FILE]
    if list_of_files:
        model_path = list_of_files[0]
        model = joblib.load(model_path)
    else:
        if not os.path.exists(TARGET_FILE):
            mr = project.get_model_registry()
            # get best model based on custom metrics
            model = mr.get_best_model(model_name,
                                      evaluation_metric,
                                      sort_metrics_by)
            model_dir = model.download()
            model = joblib.load(model_dir + "/model.pkl")

    return model
