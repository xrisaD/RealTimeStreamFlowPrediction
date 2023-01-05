import streamlit as st
import hopsworks
import pandas as pd
import folium
from streamlit_folium import folium_static
from branca.element import Figure
from PIL import Image



def fancy_header(text, font_size=24):
    res = f'<span style="color:#ff5f27; font-size: {font_size}px;">{text}</span>'
    st.markdown(res, unsafe_allow_html=True)


st.title('Streamflow Project💦')

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
st.write(36 * "-")

# Connect
fancy_header('\n📡 Connecting to Hopsworks...')

project = hopsworks.login()
dataset_api = project.get_dataset_api()

st.write("Successfully connected!✔️")
progress_bar.progress(20)

# Get Data
fancy_header('\n☁️ Getting data from Hopsworks...')

dataset_api.download("Resources/images/predictionsAbisko.png", overwrite=True)
dataset_api.download("Resources/images/predictionsSpånga.png", overwrite=True)
dataset_api.download("Resources/images/predictionsUppsala.png", overwrite=True)

dataset_api.download("Resources/latest_historical_data.csv", overwrite=True)

data_to_display = pd.read_csv('latest_historical_data.csv')

st.write("Successfully got the data!✔️")
progress_bar.progress(60)

st.write(36 * "-")
fancy_header(f"🗺 Processing the map...")

fig = Figure(width=550, height=350)

my_map = folium.Map(location=[58, 20], zoom_start=3.71)
fig.add_child(my_map)
folium.TileLayer('Stamen Terrain').add_to(my_map)
folium.TileLayer('Stamen Toner').add_to(my_map)
folium.TileLayer('Stamen Water Color').add_to(my_map)
folium.TileLayer('cartodbpositron').add_to(my_map)
folium.TileLayer('cartodbdark_matter').add_to(my_map)
folium.LayerControl().add_to(my_map)

data_to_display = data_to_display[["place", "precipitation_sum", "streamflow"]]

cities_coords = {("Abisko", "Sweden"): [68.35, 18.82],
                 ("Uppsala", "Sweden"): [59.87, 17.60],
                 ("Spånga", "Sweden"): [58.00, 12.73]}

data_to_display = data_to_display.set_index("place")

cols_names_dict = {"precipitation_sum": "Precipitation Sum",
                   "rain_sum": "Rain Sum",
                   "snowfall_sum": "Snowfall Sum",
                   "temperature_2m_max": "Temperature Max",
                   "temperature_2m_min": "Temperature Min",
                   "streamflow": "Streamflow"}

data_to_display = data_to_display.rename(columns=cols_names_dict)

for city, country in cities_coords:
    text = f"""
            <h4 style="color:green;">{city}</h4>
            <h5 style="color":"green">
                <table style="text-align: right;">
                    <tr>
                        <th>Country:</th>
                        <td><b>{country}</b></td>
                    </tr>
                    """
    for column in data_to_display.columns:
        text += f"""
                    <tr>
                        <th>{column}:</th>
                        <td>{data_to_display.loc[city][column]}</td>
                    </tr>"""
    text += """</table>
                    </h5>"""

    folium.Marker(
        cities_coords[(city, country)], popup=text, tooltip=f"<strong>{city}</strong>"
    ).add_to(my_map)

# call to render Folium map in Streamlit
folium_static(my_map)
progress_bar.progress(80)
st.sidebar.write("-" * 36)

image = Image.open('predictionsAbisko.png')
st.sidebar.image(image)

image = Image.open('predictionsSpånga.png')
st.sidebar.image(image)
image = Image.open('predictionsUppsala.png')
st.sidebar.image(image)

progress_bar.progress(100)

st.button("Re-run")