import streamlit as st
from datetime import datetime
import pandas as pd
from urllib import request, parse
import json
import pydeck as pdk

st.set_page_config(page_title="NYC Taxi Fare Predictor", page_icon="🚕", layout="wide")

st.title("🚕 NYC Taxi :yellow[Fare Predictor]")
st.markdown("Get instant :yellow[fare predictions] for your NYC taxi ride !")

url = 'https://taxifare.lewagon.ai/predict'

# Main
st.header("📍 Please give us your ride info")


# Pickup
st.subheader(" Were is the lift off 🚀 ?")
pickup_datetime = st.date_input("Date", datetime.now())
pickup_time = st.time_input("Time", datetime.now().time())
col1, col2 = st.columns(2)
pickup_longitude = col1.number_input("Pickup Longitude", value=-73.985428, format="%.6f")
pickup_latitude = col2.number_input("Pickup Latitude", value=40.748817, format="%.6f")

# Dropoff
st.subheader("Where is the touch down 🛬 ?")
col3, col4 = st.columns(2)
dropoff_longitude = col3.number_input("Dropoff Longitude", value=-73.985428, format="%.6f")
dropoff_latitude = col4.number_input("Dropoff Latitude", value=40.748817, format="%.6f")
passenger_count = st.number_input("Number of Passengers", min_value=1, max_value=10, value=1)

pickup_datetime_str = f"{pickup_datetime} {pickup_time}"

# Map
st.header("🗺️ Here's your route")

# Create points for pickup and dropoff
points_df = pd.DataFrame({
    'lat': [pickup_latitude, dropoff_latitude],
    'lon': [pickup_longitude, dropoff_longitude],
    'color': [[0, 255, 0, 180], [255, 0, 0, 180]],  # Green for pickup, Red for dropoff
    'size': [200, 200]
})

# Get real route from OSRM API
@st.cache_data
def get_route(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat):
    """Fetch actual route from OSRM routing service"""
    try:
        osrm_url = f"http://router.project-osrm.org/route/v1/driving/{pickup_lon},{pickup_lat};{dropoff_lon},{dropoff_lat}?overview=full&geometries=geojson"
        response = request.urlopen(osrm_url)
        data = json.loads(response.read())

        if data.get('code') == 'Ok' and 'routes' in data:
            # Get the coordinates from the route geometry
            coordinates = data['routes'][0]['geometry']['coordinates']
            return coordinates
        else:
            # Fallback to straight line if routing fails
            return [[pickup_lon, pickup_lat], [dropoff_lon, dropoff_lat]]
    except:
        # Fallback to straight line on error
        return [[pickup_lon, pickup_lat], [dropoff_lon, dropoff_lat]]

# Get the actual route
route_coordinates = get_route(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)

# Create route line
route_df = pd.DataFrame({
    'path': [route_coordinates],
    'color': [[255, 200, 0, 200]]  # Yellow/orange for route
})

# Calculate center point for map view
center_lat = (pickup_latitude + dropoff_latitude) / 2
center_lon = (pickup_longitude + dropoff_longitude) / 2

# Create the pydeck map with route and points
view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=11,
    pitch=0
)

# Route layer
route_layer = pdk.Layer(
    'PathLayer',
    route_df,
    get_path='path',
    get_color='color',
    width_scale=20,
    width_min_pixels=3,
    get_width=5
)

# Points layer
points_layer = pdk.Layer(
    'ScatterplotLayer',
    points_df,
    get_position='[lon, lat]',
    get_color='color',
    get_radius='size',
    pickable=True
)

deck = pdk.Deck(
    layers=[route_layer, points_layer],
    initial_view_state=view_state,
    map_style='https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    tooltip={'text': '🟢 Pickup → 🔴 Dropoff'}
)

st.pydeck_chart(deck)

# Prediction
st.header("💰 Get your  :yellow[Fare Prediction] now")

button_clicked = st.button("🚀 :yellow[Predict Fare]", type="primary", use_container_width=True)

if button_clicked:
    params = {
        "pickup_datetime": pickup_datetime_str,
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "passenger_count": passenger_count
    }

    st.spinner("🔮 Predicting fare...")

    try:
        query_string = parse.urlencode(params)
        full_url = f"{url}?{query_string}"

        response = request.urlopen(full_url)

        response_data = response.read()
        prediction = json.loads(response_data)

        if "fare" in prediction:
            fare = prediction["fare"]

            st.success("✅ Prediction successful!")

            col1, col2, col3 = st.columns([1, 2, 1])
            col2.metric(
                label="Estimated Fare",
                value=f"${fare:.2f}",
                delta=None
            )

            st.info(f"💡 This is an estimated fare based on the route parameters. Actual fare may vary based on traffic, route taken, and other factors.")

        else:
            st.error("❌ Unexpected response format from API")
            st.json(prediction)

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

st.markdown("---")
st.markdown("Made with ❤️ and a lot of ☕️ using Streamlit and the TaxiFare API from Le Wagon", width="stretch")
