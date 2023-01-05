import pandas as pd

from datetime import datetime

from ml_system.features import feature_engineering

place_list = ['Abisko', 'Uppsala', 'Spånga']

place_streamflow = [2357, 2609, 2212]

lat_long = [[68.35, 18.82], [59.87, 17.60], [58.00, 12.73]]


def test_yesterday():
    today = datetime(2020, 5, 17)
    yesterday = datetime(2020, 5, 16)
    res_yesterday = feature_engineering.get_yesterday(today)
    assert yesterday == res_yesterday


def test_days_before():
    today = datetime(2020, 5, 17)
    other_date = datetime(2020, 5, 15)
    res_other_date = feature_engineering.get_prev_days(today, 2)
    assert other_date == res_other_date


def test_yesterday_no_time():
    today = datetime(2020, 5, 17, 13, 30, 0)
    yesterday = datetime(2020, 5, 16)
    res_yesterday = feature_engineering.get_yesterday(today)
    assert yesterday == res_yesterday


def test_timestamp_2_time_no_time():
    today1 = feature_engineering.timestamp_2_time(datetime(2020, 5, 17))
    today2 = feature_engineering.timestamp_2_time(datetime(2020, 5, 17, 13, 30, 0))

    assert today1 == today2


def test_timestamp_2_time_no_time():
    today1 = feature_engineering.timestamp_2_time(datetime(2020, 5, 17))
    today2 = feature_engineering.timestamp_2_time(datetime(2020, 5, 17, 13, 30, 0))

    assert today1 == today2


def test_get_streamflow_data():
    today = datetime(2020, 5, 17)
    data = feature_engineering.get_streamflow_data(place_streamflow[0], today, place_list[0])

    assert len(data[0]) == 3
    assert data[0][2] == place_list[0]


def test_get_weather_data():
    col_names = ['time', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'rain_sum', 'snowfall_sum',
                 'precipitation_hours', 'windspeed_10m_max', 'windgusts_10m_max', 'winddirection_10m_dominant',
                 'et0_fao_evapotranspiration', 'place']

    data = feature_engineering.get_weather_data(lat_long[0], place_list[0])

    assert len(data.keys()) == 12
    for k in col_names:
        assert k in data.keys()
    assert len(data[col_names[0]]) == 8
    assert data["place"] == place_list[0]


def test_get_streamflow_df():
    data = [['2020-05-17', '1', 'Abisko'], ['2020-05-17', '2', 'Uppsala']]
    df = feature_engineering.get_streamflow_df(data)
    assert type(df) == pd.DataFrame
    assert df.size == 6
    assert len(df.columns) == 3
    assert len(df.index) == 2


def test_get_weather_df():
    col_names = ['date', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'rain_sum', 'snowfall_sum',
                 'precipitation_hours', 'windspeed_10m_max', 'windgusts_10m_max', 'winddirection_10m_dominant',
                 'et0_fao_evapotranspiration', 'place']

    data = [
        {'time': ['2022-12-25', '2022-12-26'], 'temperature_2m_max': [-3.8, -6.7], 'temperature_2m_min': [-10.2, -11.9],
         'precipitation_sum': [0.0, 0.0], 'rain_sum': [0.0, 0.0], 'snowfall_sum': [0.0, 0.0],
         'precipitation_hours': [0.0, 0.0], 'windspeed_10m_max': [20.9, 13.0], 'windgusts_10m_max': [40.7, 41.4],
         'winddirection_10m_dominant': [224, 46], 'et0_fao_evapotranspiration': [0.18, 0.03], 'place': 'Abisko'}]

    df = feature_engineering.get_weather_df(data)

    assert type(df) == pd.DataFrame
    assert df.size == 24
    assert len(df.columns) == 12
    assert len(df.index) == 2

    for k in col_names:
        assert k in df.columns

