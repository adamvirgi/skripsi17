import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model and scaler
model = joblib.load('model_svm.joblib')
scaler = joblib.load('scaler_minmax.joblib')

# Define the application title and layout
st.title('Status Gizi Balita')
st.write('Aplikasi ini memprediksi status gizi balita berdasarkan usia, jenis kelamin, dan tinggi badan.')

# Get user input
age = st.number_input('Usia (bulan)', min_value=0, max_value=60)
gender = st.selectbox('Jenis Kelamin', ['Perempuan', 'Laki-laki'])
height = st.number_input('Tinggi Badan (cm)', min_value=0, max_value=120)

gender_encoded = 0 if gender == 'Perempuan' else 1

# Preprocess the user input
user_input = np.array([[age, gender, height]])
user_input_scaled = scaler.transform(user_input)

# Predict the status
prediction = model.predict(user_input_scaled)[0]

# Display the prediction
if st.button('Prediksi'):
    if prediction == 'stunted':
        st.write('Status gizi balita: Stunting')
    elif prediction == 'tall':
        st.write('Status gizi balita: Tinggi')
    elif prediction == 'normal':
        st.write('Status gizi balita: Normal')
    else:
        st.write('Status gizi balita: Sangat Pendek')

