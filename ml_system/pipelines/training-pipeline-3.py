import sys
from datetime import datetime

import modal
from hsfs.feature import Feature

sys.path.append('../../')
from ml_system.features.feature_engineering import *
from ml_system.features.inference import *

place_list = ['Abisko', 'Uppsala', 'Spånga']

place_streamflow = [2357, 2609, 2212]

lat_long = [[68.35, 18.82], [59.87, 17.60], [58.00, 12.73]]

BACKFILL = False
LOCAL = False

if LOCAL == False:
    stub = modal.Stub("train-xgboost")
    image = modal.Image.debian_slim().pip_install(["hopsworks", "joblib", "scikit-learn", "xgboost"])


    @stub.function(image=image, schedule=modal.Period(days=7), timeout=1000,
                   secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


def train_model(pipe, train_data):
    from sklearn.metrics import f1_score

    X = shift_data(train_data)
    y = X.pop("streamflow_next_day")

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
    # get latest data
    start_date = get_prev_days(datetime.today(), 7)
    start_date = timestamp_2_time(start_date)
    query = streamflow_fg.filter(Feature("date") >= int(start_date)).join(weather_fg.filter(Feature("date") >= int(start_date)))

    feature_view = fs.create_feature_view(
        name='streamflow_fv',
        query=query
    )

    feature_view.create_training_data()
    train_data = feature_view.get_training_data(1)[0]

    train_data = train_data.sort_values(by=["date", 'place'], ascending=[False, True]).reset_index(drop=True)

    model = get_model(project=project,
                      model_name="xgb_pipeline",
                      evaluation_metric="f1",
                      sort_metrics_by="max")

    # retrain the model with the new data
    pipe, f1, X, y = train_model(model, train_data)

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
