import sys
import modal

sys.path.append('../../')

from ml_system.features.inference import shift_data

place_list = ['Abisko', 'Uppsala', 'Spånga']

place_streamflow = [2357, 2609, 2212]

lat_long = [[68.35, 18.82], [59.87, 17.60], [58.00, 12.73]]

BACKFILL = False
LOCAL = False

if LOCAL == False:
    stub = modal.Stub("train-xgboost")
    image = modal.Image.debian_slim().pip_install(["hopsworks", "joblib", "scikit-learn", "xgboost"])


    @stub.function(image=image, schedule=modal.Period(days=7), timeout=1000, secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


    @stub.function(image=image, timeout=600, secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def fit_xgboost(data):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.metrics import f1_score
        import xgboost as xgb

        params = data[0]
        train_data = data[1]
        test_data = data[2]
        X = shift_data(train_data)
        y = X.pop("streamflow_next_day")

        column_trans = ColumnTransformer(
            [('standard_scaler', StandardScaler(),
              ['streamflow', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'rain_sum',
               'snowfall_sum',
               'precipitation_hours', 'windspeed_10m_max', 'windgusts_10m_max', 'winddirection_10m_dominant',
               'et0_fao_evapotranspiration'])
             ],
            verbose_feature_names_out=True)
        xgb = xgb.XGBRegressor(**params)
        pipe = Pipeline([('transformations', column_trans), ('xgb', xgb)])

        pipe.fit(X, y)

        # evaluate on the test data
        X2 = test_data.drop(columns=["date", "place"]).fillna(0)
        y2 = test_data.pop("streamflow")

        f1 = f1_score(y2.astype('int'), [int(pred) for pred in pipe.predict(X2)], average='micro')

        return f1, params


def train_model(train_data, params):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import f1_score
    import xgboost as xgb

    X = shift_data(train_data)
    y = X.pop("streamflow_next_day")

    column_trans = ColumnTransformer(
        [('standard_scaler', StandardScaler(),
          ['streamflow', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'rain_sum',
           'snowfall_sum',
           'precipitation_hours', 'windspeed_10m_max', 'windgusts_10m_max', 'winddirection_10m_dominant',
           'et0_fao_evapotranspiration'])
         ],
        verbose_feature_names_out=True)
    pipe = Pipeline([('transformations', column_trans), ('xgb', xgb.XGBRegressor(max_depth=params['max_depth'], n_estimators=params['n_estimators']))])

    pipe.fit(X, y)

    f1 = f1_score(y.astype('int'), [int(pred) for pred in pipe.predict(X)], average='micro')

    return pipe, f1, X, y


def g():
    import hopsworks
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    import joblib

    # Get the feature groups and create the feature view
    project = hopsworks.login()
    fs = project.get_feature_store()

    streamflow_fg = fs.get_or_create_feature_group(
        name='streamflow_fg',
        version=1
    )
    weather_fg = fs.get_or_create_feature_group(
        name='weather_fg',
        version=1
    )
    query = streamflow_fg.select_all().join(weather_fg.select_all())

    feature_view = fs.create_feature_view(
        name='streamflow_fv',
        query=query
    )

    feature_view.create_training_data()
    train_data = feature_view.get_training_data(1)[0]

    train_data = train_data.sort_values(by=["date", 'place'], ascending=[False, True]).reset_index(drop=True)

    # train-test split
    test_size = 0.15
    last_k = int(len(train_data.index) * test_size)
    test_data = train_data[-last_k:-1]
    train_data2 = train_data[:len(train_data.index) - last_k]

    # hyperparameter tuning
    gbm_param_grid = [{'n_estimators': 50, 'max_depth': 2}, {'n_estimators': 50, 'max_depth': 5},
                      {'n_estimators': 100, 'max_depth': 2}, {'n_estimators': 100, 'max_depth': 5},
                      {'n_estimators': 100, 'max_depth': None}]

    data = [(params, train_data2, test_data) for params in gbm_param_grid]
    best_score, best_params = max(fit_xgboost.map(data))

    # retrain the model with the best parameters on the whole dataset
    pipe, f1, X, y = train_model(train_data, best_params)

    # save the model
    mr = project.get_model_registry()

    input_schema = Schema(X)
    output_schema = Schema(y)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    model_schema.to_dict()

    joblib.dump(pipe, './model.pkl')

    model = mr.sklearn.create_model(
        name="xgb_pipeline",
        metrics={"f1": f1},
        description="Tranformations and XGBoost Regressor.",
        input_example=train_data.sample(),
        model_schema=model_schema
    )

    model.save('model.pkl')


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        with stub.run():
            f()
