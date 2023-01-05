import pandas as pd


from ml_system.features.inference import shift_data


def test_shift_data():
    d = {'date': ['2020-05-17', '2020-05-18', '2020-05-19', '2020-05-20'],
         'place': ['Abisko', 'Abisko', 'Abisko', 'Abisko'], 'streamflow': [1, 2, 3, 4], 'weather': [10, 11, 12, 14]}
    df = pd.DataFrame(d)
    df2 = shift_data(df)

    assert 'date' not in df2.keys()
    assert 'streamflow_next_day' in df2.keys()
    assert df2.iloc[0]['streamflow_next_day'] == 0
    assert df2.iloc[1]['streamflow_next_day'] == 4