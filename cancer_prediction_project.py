# -*- coding: utf-8 -*-
"""Cancer_prediction_project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1IK5MlPZIRiAp0nnlbS9gNSunMXOogV3S
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

filepath="/content/framingham.csv"
data=pd.read_csv(filepath)
b=data.columns
print(b)

data.isnull().sum()
imputer=SimpleImputer(strategy = "mean")
data=imputer.fit_transform(data)

data=pd.DataFrame(data,columns=b)
data.head()
data.isnull().sum()

X=data.drop("TenYearCHD",axis=1)
y=data["TenYearCHD"]

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

with open ("scaler.pkl","wb") as file:
  pickle.dump(scaler,file)

with open ("imputer.pkl","wb") as file:
  pickle.dump(imputer,file)

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.models import load_model

# Define the model
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])

# Display the model summary
model.summary()

# Set up TensorBoard logging directory
log_dir = "logs/fits/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Set up EarlyStopping callback
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=[tensorboard_callback, early_stopping_callback])

# Save the trained model
model.save("model.h5")

# %load_ext tensorboard
# %tensorboard --logdir logs/fits

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.models import load_model

model=load_model("model.h5")

with open ("scaler.pkl","rb")as file:
  scaler=pickle.load(file)

with open ("imputer.pkl","rb")as file:
  imputer=pickle.load(file)

input_data ={'male':1, 'age':27, 'education':3, 'currentSmoker':1, 'cigsPerDay':40, 'BPMeds':0,
       'prevalentStroke':0, 'prevalentHyp':0, 'diabetes':0, 'totChol':250, 'sysBP':150,
       'diaBP':85, 'BMI':30, 'heartRate':95, 'glucose':75
}

keys=list(input_data.keys())
input_data=pd.DataFrame([input_data],columns=keys)
input_data

input_data=scaler.fit_transform(input_data)
input_data

prediction=model.predict(input_data)
print(prediction)
prediction_proba=prediction[0][0]

if prediction_proba>0.5:
  print("Pls take care ,you may have heart cancer in ten years span  as per prediction on your report")
else:
  print("you may have  not heart cancer in ten years span  as per prediction on your report")

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