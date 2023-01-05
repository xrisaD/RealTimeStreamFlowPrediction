import sys

sys.path.append('../../')

from features.feature_engineering import *

import modal

place_list = ['Abisko', 'Uppsala', 'Spånga']

place_streamflow = [2357, 2609, 2212]

lat_long = [[68.35, 18.82], [59.87, 17.60], [58.00, 12.73]]

BACKFILL = False
LOCAL = False

if LOCAL == False:
    stub = modal.Stub("feature-pipeline-daily")
    image = modal.Image.debian_slim().pip_install(["hopsworks"])


    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


def get_data():
    # we get the data for yesterday and the day before yesterday
    # to be sure that we have all the data because the APIs are not updated on a specific time
    yesterday_and_the_day_before = get_prev_days(datetime.today(), 2)

    data_streamflow = []
    for p1, p2 in zip(place_streamflow, place_list):
        data_streamflow.extend(get_streamflow_data(p1, yesterday_and_the_day_before, p2))
    df_streamflow = get_streamflow_df(data_streamflow)

    data_weather = [get_weather_data(ll, place) for ll, place in zip(lat_long, place_list)]
    df_weather = get_weather_df(data_weather)
    return df_streamflow, df_weather


def g():
    import hopsworks

    # Get the feature groups and the feature view
    project = hopsworks.login()
    fs = project.get_feature_store()

    streamflow_fg = fs.get_or_create_feature_group(
        name='streamflow_fg',
        description='Streamflow characteristics of each day',
        version=1,
        primary_key=['place', 'date'],
        online_enabled=True,
        event_time='date'
    )

    weather_fg = fs.get_or_create_feature_group(
        name='weather_fg',
        description='Weather characteristics of each day',
        version=1,
        primary_key=['place', 'date'],
        online_enabled=True,
        event_time='date'
    )

    # get yesterday's data
    df_streamflow, df_weather = get_data()

    # insert the data into the feature groups

    streamflow_fg.insert(df_streamflow, write_options={"wait_for_job" : False})

    weather_fg.insert(df_weather, write_options={"wait_for_job" : False})


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        with stub.run():
            f()
