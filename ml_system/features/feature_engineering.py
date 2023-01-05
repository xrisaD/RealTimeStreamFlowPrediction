from datetime import datetime, timedelta
import requests
import csv
import pandas as pd


def timespamp_2_date(timestamp):
    return datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d")


def timestamp_2_time(x):
    if type(x) == datetime:
        x = x.strftime("%Y-%m-%d")
    dt_obj = datetime.strptime(str(x), '%Y-%m-%d')
    dt_obj = dt_obj.timestamp() * 1000
    return int(dt_obj)


# datetime.today()
def get_yesterday(the_date):
    return get_prev_days(the_date, 1)


def get_prev_days(the_date, days_to_subtract):
    the_date = (the_date - timedelta(days=days_to_subtract)).strftime("%Y-%m-%d")
    the_date = datetime.strptime(the_date, '%Y-%m-%d')
    return the_date


def get_streamflow_data(s, the_date, place):
    streamflow_url = 'https://opendata-download-hydroobs.smhi.se/api/version/latest/parameter/1/station/{}/period/corrected-archive/data.csv'.format(
        s)

    with requests.Session() as s:
        download = s.get(streamflow_url)

        decoded_content = download.content.decode('utf-8')
        cr = list(csv.reader(decoded_content.splitlines(), delimiter=';'))
        res = []
        for row in cr:
            try:
                d = datetime.strptime(row[0], '%Y-%m-%d')
                if the_date <= d:
                    res.append(row[0:2] + [place])
            except:
                pass
        return res


def get_weather_data(ll, place, past_days=1):
    weather_api = 'https://api.open-meteo.com/v1/forecast?timezone=Europe/Berlin&latitude={}&longitude={}&past_days={}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum,precipitation_hours,windspeed_10m_max,windgusts_10m_max,winddirection_10m_dominant,et0_fao_evapotranspiration'
    res = requests.get(weather_api.format(ll[0], ll[1], past_days)).json()['daily']
    res['place'] = place
    return res


def get_streamflow_df(data):
    col_names = ["date", "streamflow", "place"]

    res = pd.DataFrame(
        data,
        columns=col_names
    )
    res.date = res.date.apply(timestamp_2_time)
    res['streamflow'] = res['streamflow'].astype(float)
    return res


def get_weather_df(data_weather):
    res = []
    for dw in data_weather:
        df = pd.DataFrame.from_dict(dw)
        res.append(df)  # .drop( df.index.to_list()[1:] ,axis = 0 )
    res = pd.concat(res)
    res = res.rename(columns={"time": "date"})
    res.date = res.date.apply(timestamp_2_time)
    res['winddirection_10m_dominant'] = res['winddirection_10m_dominant'].astype(float)

    return res
