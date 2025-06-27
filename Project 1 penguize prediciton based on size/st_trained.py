import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd
data = pd.read_csv('penguins_size.csv')  

le_species = LabelEncoder()
data['island_id'] = le_species.fit_transform(data['island'])

le_sex = LabelEncoder()
data['species_id'] = le_sex.fit_transform(data['species'])

loaded_model_species = joblib.load('penguin_model_species.pkl')

loaded_model_sex = joblib.load('penguin_model_sex.pkl')


enter_culmen_length = st.slider('Culmen Length (mm)', min_value=min(data['culmen_length_mm']), max_value=max(data['culmen_length_mm']), value=data['culmen_length_mm'][0])

enter_flipper_length = st.slider('Flipper Length (mm)', min_value=min(data['flipper_length_mm']), max_value=max(data['flipper_length_mm']), value=data['flipper_length_mm'][0])

enter_island = st.selectbox('Island', options=data['island'].unique(), index=0)
island_id = le_species.transform([enter_island])[0]


# plot data for culemn length and flipper length and with slider also show current positon of poitns and st the model prediction_species
st.write("Current Data Provided:")
st.write(f"Culmen Length: {enter_culmen_length} mm,  Flipper Length: {enter_flipper_length} mm,  Island: {enter_island} (ID: {island_id})")
st.write("Model Prediction:")
prediction_species = loaded_model_species.predict([[enter_culmen_length, enter_flipper_length, island_id]])

species_id = le_sex.transform([prediction_species])[0]

prediction_sex = loaded_model_sex.predict([[enter_culmen_length, species_id]])

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(data['culmen_length_mm'], data['flipper_length_mm'], c=data['species'].astype('category').cat.codes, cmap='viridis', alpha=0.5)
ax.scatter(enter_culmen_length, enter_flipper_length, color='red', s=100, marker='x', label=f'Species:{prediction_species[0]} \nGender: {prediction_sex[0]}')
ax.set_xlabel('Culmen Length (mm)')
ax.set_ylabel('Flipper Length (mm)')
ax.set_title('Penguin Species by Culmen and Flipper Length')
# fix axis limits
ax.set_xlim(min(data['culmen_length_mm']) - 5, max(data['culmen_length_mm']) + 5)
ax.set_ylim(min(data['flipper_length_mm']) - 5, max(data['flipper_length_mm']) + 5)
ax.legend()
st.pyplot(fig)

st.write(f"**Predicted Species: {prediction_species[0]}**")
st.write(f"**Predicted Gender: {prediction_sex[0]}**")

