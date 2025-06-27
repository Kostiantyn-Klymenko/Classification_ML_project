import streamlit as st
import joblib
import pandas as pd
import numpy as np

data_gender = pd.read_csv('corrected_data.csv')  

loaded_model_gender = joblib.load('gender_model.pkl')


enter_diameter = st.slider('Diameter', min_value=min(data_gender['Diameter']), max_value=max(data_gender['Diameter']), value=np.mean(data_gender['Diameter'][0]))

enter_shucked_weight = st.slider('Shucked Weight', min_value=min(data_gender['ShuckedWeight']), max_value=max(data_gender['ShuckedWeight']), value=data_gender['ShuckedWeight'][0])


st.write("Current Data Provided:")
st.write(f"Diameter: {enter_diameter},  Shucked Weight: {enter_shucked_weight}")
st.write("Model Prediction:")
prediction_gender = loaded_model_gender.predict([[enter_diameter, enter_shucked_weight]])


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(data_gender['Diameter'], data_gender['ShuckedWeight'], c=data_gender['species'].astype('category').cat.codes, cmap='viridis', alpha=0.5)
ax.scatter(enter_diameter, enter_shucked_weight, color='red', s=100, marker='x', label=f'Gender: {prediction_gender[0]}')
ax.set_xlabel('Diameter')
ax.set_ylabel('Shucked Weight')
ax.set_title('Penguin Gender by Diameter and Shucked Weight')
# fix axis limits
ax.set_xlim(min(data_gender['Diameter']) - 5, max(data_gender['Diameter']) + 5)
ax.set_ylim(min(data_gender['ShuckedWeight']) - 5, max(data_gender['ShuckedWeight']) + 5)
ax.legend()
st.pyplot(fig)

st.write(f"**Predicted Gender: {prediction_gender[0]}**")
