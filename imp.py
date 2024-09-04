
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.models import load_model


model=load_model("model.h5")

with open ("scaler.pkl","rb") as file:
  scaler=pickle.load(file)

with open ("imputer.pkl","rb") as file:
  imputer=pickle.load(file) 

st.title("10-year Risk of coronary heart disease Prediction")


male = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
age = st.slider("Age", min_value=0, max_value=120, value=27)
education = st.slider("Education Level", min_value=1, max_value=4, value=3)
currentSmoker = st.selectbox("Current Smoker", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
cigsPerDay = st.slider("Cigarettes per Day", min_value=0, max_value=100, value=40)
BPMeds = st.selectbox("Blood Pressure Medications", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
prevalentStroke = st.selectbox("Prevalent Stroke", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
prevalentHyp = st.selectbox("Prevalent Hypertension", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
diabetes = st.selectbox("Diabetes", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
totChol = st.slider("Total Cholesterol", min_value=100, max_value=500, value=250)
sysBP = st.slider("Systolic Blood Pressure", min_value=80, max_value=200, value=150)
diaBP = st.slider("Diastolic Blood Pressure", min_value=50, max_value=120, value=85)
BMI = st.slider("Body Mass Index", min_value=10, max_value=50, value=30)
heartRate = st.slider("Heart Rate", min_value=40, max_value=200, value=95)
glucose = st.slider("Glucose Level", min_value=50, max_value=200, value=75)

input_data = pd.DataFrame({
    'male': [male],
    'age': [age],
    'education': [education],
    'currentSmoker': [currentSmoker],
    'cigsPerDay': [cigsPerDay],
    'BPMeds': [BPMeds],
    'prevalentStroke': [prevalentStroke],
    'prevalentHyp': [prevalentHyp],
    'diabetes': [diabetes],
    'totChol': [totChol],
    'sysBP': [sysBP],
    'diaBP': [diaBP],
    'BMI': [BMI],
    'heartRate': [heartRate],
    'glucose': [glucose]
})

input_data_scaled=scaler.fit_transform(input_data)

prediction=model.predict(input_data_scaled)
prediction_proba=prediction[0][0]

st.write("Prediction Probability")

if prediction_proba>0.5:
  st.write("Pls take care ,you may have heart cancer in ten years span  as per prediction on your report")
else:
  st.write("you may have  not heart cancer in ten years span  as per prediction on your report")





