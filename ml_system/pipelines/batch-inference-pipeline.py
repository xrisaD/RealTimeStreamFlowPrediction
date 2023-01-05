import modal
import sys
from hsfs.feature import Feature

sys.path.append('../../')

from ml_system.features.inference import *
from ml_system.features.feature_engineering import *

place_list = ['Abisko', 'Uppsala', 'Spånga']

place_streamflow = [2357, 2609, 2212]

lat_long = [[68.35, 18.82], [59.87, 17.60], [58.00, 12.73]]

cities_coords = {("Abisko", "Sweden"): [68.35, 18.82],
                 ("Uppsala", "Sweden"): [59.87, 17.60],
                 ("Spånga", "Sweden"): [58.00, 12.73]}

BACKFILL = False
LOCAL = False

if LOCAL == False:
    stub = modal.Stub("batch-inference-pipeline")
    image = modal.Image.debian_slim().pip_install(["hopsworks", "kaleido", "matplotlib", "plotly", "scikit-learn", "xgboost"])


    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


def get_predictions(X, model, weather_df):
    cities = [city_tuple[0] for city_tuple in cities_coords.keys()]

    df_final = []
    X_test = X
    for p in range(0, 6):
        # get predictions
        preds = model.predict(X_test)

        # print the prediction for that day
        next_day_date = datetime.today() + timedelta(days=p)
        next_day = next_day_date.strftime('%Y-%m-%d')
        # add df
        df = pd.DataFrame(data=list(zip(cities, preds)), columns=["place", "streamflow_pred"], dtype=int)
        df['date'] = next_day
        df_final.append(df)

        next_day_timestamp = timestamp_2_time(next_day_date)
        cur_weather_df = weather_df.loc[weather_df["date"] == next_day_timestamp].sort_values(by=["place"])
        cur_weather_df['streamflow'] = preds
        X_test = cur_weather_df

    df_final = pd.concat(df_final)

    return df_final


def get_future_data(streamflow_fg, weather_fg):
    # get start date
    start_date = get_prev_days(datetime.today(), 1)  # TODO: change to 1 or have 2
    # get the timestamp for that date
    start_date = timestamp_2_time(start_date)
    # get the data after that date
    weather_data = weather_fg.filter(Feature("date") >= int(start_date)).read()
    streamflow_data = streamflow_fg.filter(Feature("date") == int(start_date)).read()
    # join the data based on the date
    X = weather_data.merge(streamflow_data, on=['date', 'place'])
    X = X.fillna(0).sort_values(by=["place"])
    return X, weather_data


def get_history_data(monitor_fg, weather_fg, streamflow_fg):
    start_date = timestamp_2_time(get_prev_days(datetime.today(), 10))
    end_date = timestamp_2_time(get_prev_days(datetime.today(), 0))

    weather_data = weather_fg.filter(end_date >= Feature("date") >= int(start_date)).read()
    monitor_data = monitor_fg.filter(end_date >= Feature("date") >= int(start_date)).read()
    streamflow_data = streamflow_fg.filter(end_date >= Feature("date") >= int(start_date)).read()

    X1 = weather_data.merge(streamflow_data, on=['date', 'place'])
    X2 = X1.merge(monitor_data, on=['date', 'place'])
    #X1.date = X1.date.apply(timespamp_2_date)
    X2.date = X2.date.apply(timespamp_2_date)
    return X1, X2


def get_df_image(df):
    import plotly.figure_factory as ff
    df = df.loc[:, ["date", "streamflow", "streamflow_pred", "precipitation_sum", "rain_sum"]].rename(columns={"date": "Date", "streamflow":"Streamflow","streamflow_pred": "Prediction","precipitation_sum": "Precipitation", "rain_sum":"Rain"})
    fig = ff.create_table(df)

    return fig

def g():
    import hopsworks
    import matplotlib.pyplot as plt
    # Get the feature groups and the feature view
    project = hopsworks.login()
    fs = project.get_feature_store()
    dataset_api = project.get_dataset_api()

    # get feature groups
    streamflow_fg = fs.get_or_create_feature_group(
        name='streamflow_fg',
        version=1
    )



    weather_fg = fs.get_or_create_feature_group(
        name='weather_fg',
        version=1
    )

    monitor_fg = fs.get_or_create_feature_group(name="streamflow_predictions",
                                                version=1,
                                                primary_key=["date", "place"],
                                                description="Streamflow Prediction/Outcome Monitoring"
                                                )

    model = get_model(project=project,
                      model_name="xgb_pipeline",
                      evaluation_metric="f1",
                      sort_metrics_by="max")

    X, weather_data = get_future_data(streamflow_fg, weather_fg)


    predictions_df = get_predictions(X, model, weather_data)


    # save the predictions
    predictions_df2 = predictions_df.copy()
    predictions_df2.date = predictions_df2.date.apply(timestamp_2_time)


    monitor_fg.insert(predictions_df2, write_options={"wait_for_job": False})

    # images with history and predictions
    history_df1, history_df2 = get_history_data(monitor_fg, weather_fg, streamflow_fg)

    for p in place_list:
        # image for future predictions
        df_tmp = predictions_df.loc[predictions_df['place'] == p]

        plt.plot(df_tmp["date"], df_tmp["streamflow_pred"], 'o-r')
        plt.ylabel('Streamflow Prediction')
        plt.xlabel('Date')
        plt.title('Predictions for {}'.format(p))
        # plt.show()

        name = 'predictions{}.png'.format(p)
        plt.savefig(name)

        dataset_api.upload("./" + name, "Resources/images", overwrite=True)
        plt.clf()


        # image for historical data
        df_tmp = history_df2.loc[history_df2['place'] == p]
        df_tmp = df_tmp.sort_values(by=["date"])
        fig = get_df_image(df_tmp.tail(5))
        name = 'history{}.png'.format(p)
        fig.write_image(name)
        dataset_api.upload("./" + name, "Resources/images", overwrite=True)

    # save the latest historical data
    latest_data = pd.DataFrame()
    days = 0
    while latest_data.empty:
        date = timestamp_2_time(get_prev_days(datetime.today(), days))
        latest_data = history_df1.loc[history_df1["date"] == date]
        days = days + 1

    name = 'latest_historical_data.csv'
    latest_data.to_csv(name)
    dataset_api.upload("./" + name, "Resources", overwrite=True)



if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        with stub.run():
            f()
