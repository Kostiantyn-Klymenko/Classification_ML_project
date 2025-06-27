import streamlit as st
import joblib
import pandas as pd
import numpy as np

data_gender = pd.read_csv('corrected_data.csv')  

loaded_model_gender = joblib.load('svm_model.pkl')

st.title(" Preduction for Gender based on many features")


# Create input fields for the features

#svm_model.predict([[ 0.74  ,  0.595 ,  0.19  ,  2.3235,  1.1495,  0.5115,  0.505 ,11.    ]])

st.form("Enter the features for prediction:")
feature1 = st.number_input("Feature 1", value=0.74)
feature2 = st.number_input("Feature 2", value=0.595)
feature3 = st.number_input("Feature 3", value=0.19)
feature4 = st.number_input("Feature 4", value=2.3235)
feature5 = st.number_input("Feature 5", value=1.1495)
feature6 = st.number_input("Feature 6", value=0.5115)
feature7 = st.number_input("Feature 7", value=0.505)
feature8 = st.number_input("Feature 8", value=11.0)
if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8]])
    prediction = loaded_model_gender.predict(input_data)
    st.write(f"Predicted result: {prediction[0]}")



