import sys
import modal

sys.path.append('../../')
from features.inference import shift_data

place_list = ['Abisko', 'Uppsala', 'Spånga']

place_streamflow = [2357, 2609, 2212]

lat_long = [[68.35, 18.82], [59.87, 17.60], [58.00, 12.73]]

BACKFILL = False
LOCAL = False

if LOCAL == False:
    stub = modal.Stub("train-gradient-boost-regressor")
    image = modal.Image.debian_slim().pip_install(["hopsworks", "joblib", "scikit-learn"])


    @stub.function(image=image, schedule=modal.Period(days=15), timeout=600, secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


def g():
    import hopsworks
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import f1_score

    # Get the feature groups and the feature view
    project = hopsworks.login()
    fs = project.get_feature_store()

    # get the data
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
        #version=1,
        query=query
    )



    feature_view.create_training_data()

    train_data = feature_view.get_training_data(1)[0]


    X = shift_data(train_data)

    y = X.pop("streamflow_next_day")

    column_trans = ColumnTransformer(
        [('standard_scaler', StandardScaler(),
             ['streamflow', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'rain_sum', 'snowfall_sum',
              'precipitation_hours', 'windspeed_10m_max', 'windgusts_10m_max', 'winddirection_10m_dominant',
              'et0_fao_evapotranspiration'])
        ],
        verbose_feature_names_out=True)
    pipe = Pipeline([('transformations', column_trans), ('gb', GradientBoostingRegressor())])

    pipe.fit(X, y)

    f1 = f1_score(y.astype('int'), [int(pred) for pred in pipe.predict(X)], average='micro')

    mr = project.get_model_registry()

    input_schema = Schema(X)
    output_schema = Schema(y)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    model_schema.to_dict()

    joblib.dump(pipe, './model.pkl')

    model = mr.sklearn.create_model(
        name="gradient_boost_pipeline",
        metrics={"f1": f1},
        description="Tranformations and Gradient Boost Regressor.",
        input_example=X.sample(),
        model_schema=model_schema
    )

    model.save('model.pkl')


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        with stub.run():
            f()
