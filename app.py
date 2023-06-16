import datetime as dt

from joblib import load
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_data
def get_data():
    df = pd.read_csv("dataset/cleaned.csv")

    return df

@st.cache_data
def get_encoders():
    categorical_encoder = load("artifacts/categorical_encoder.joblib")
    scaler = load("artifacts/scaler.joblib")

    return categorical_encoder, scaler

@st.cache_data
def get_model():
    model = load("artifacts/lightgbm.joblib")

    return model

df = get_data()
categorical_encoder, scaler = get_encoders()
model = get_model()

st.markdown("# Flight Ticket Price Prediction")

date = st.date_input(label="Date")

col1, col2, col3 = st.columns(3)
departure_date = col1.date_input(label="Departure Date")
departure_time = col2.time_input(label="Departure Time",step=300)
duration = col3.number_input(label="Duration (Minutes)", min_value=1, step=5, value=30)

days_left = (departure_date - date).days

if days_left < 0:
    st.error("The departure date cannot be before the booking date.")
    st.stop()

departure_time_dt = dt.datetime.combine(dt.date(1, 1, 1), departure_time)
arrival_date = departure_time_dt + dt.timedelta(minutes=duration)
arrival_time = arrival_date.time()

col1, col2, col3 = st.columns(3)
flight_class = col1.selectbox(label="Class", options=df["class"].unique())
airline = col2.selectbox(label="Airline", options=df["airline"].unique())
flight = col3.selectbox(label="Flight", options=np.concatenate((["UNK"], df["flight"].unique())))

col1, col2 = st.columns(2)
source_city = col1.selectbox(label="Source City", options=df["source_city"].unique())
destination_city = col2.selectbox(label="Destination City", options=df["destination_city"].unique())

if source_city == destination_city:
    st.error("Source and destination cities cannot be the same.")
    st.stop()

col1, col2 = st.columns(2)
stops = col1.selectbox(label="Stops", options=df["stops"].unique())
via = col2.selectbox(label="Stopover Location", options=df["via"].unique())

categorical_inputs = [
    date,
    airline,
    flight,
    source_city,
    destination_city,
    stops,
    via,
    flight_class,
]

categorical_inputs_t = categorical_encoder.transform([categorical_inputs])

numerical_inputs = [
    days_left,
    departure_time.hour,
    departure_time.minute,
    arrival_time.hour,
    arrival_time.minute,
    duration
]

inputs = np.hstack((categorical_inputs_t, [numerical_inputs]))
inputs_t = scaler.transform(inputs)

conversion_rate = 0.056
prediction = np.expm1(model.predict(inputs_t))[0]

st.info(f"INR{prediction:.2f} (≈ RM{prediction * conversion_rate:.2f})")
