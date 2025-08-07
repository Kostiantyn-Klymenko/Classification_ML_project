import streamlit as st
import joblib
import pandas as pd




# Load the dataset
data = pd.read_csv(r'C:\Users\agent\Downloads\Classification_ML_project\Project 3 Regression with SVM\cement_slump.csv')
# Load the trained model
loaded_model = joblib.load(r'C:\Users\agent\Downloads\Classification_ML_project\Project 3 Regression with SVM\cement_svr_model.pkl')


st.title("Cement Slump Prediction")
# column names are Cement,Slag,Fly ash,Water,SP,Coarse Aggr.,Fine Aggr.,SLUMP(cm),FLOW(cm
st.form("Enter the features for prediction:")
feature1 = st.slider("Cement", min_value=0.0, max_value=500.0, value=300.0)
feature2 = st.slider("Slag", min_value=0.0, max_value=200.0, value=50.0)
feature3 = st.slider("Fly ash", min_value=0.0, max_value=200.0, value=50.0)     
feature4 = st.slider("Water", min_value=0.0, max_value=200.0, value=150.0)
feature5 = st.slider("SP", min_value=0.0, max_value=10.0, value=2.0)
feature6 = st.slider("Coarse Aggr.", min_value=0.0, max_value=1000.0, value=800.0)
feature7 = st.slider("Fine Aggr.", min_value=0.0, max_value=1000.0, value=600.0)
feature8 = st.slider("SLUMP (cm)", min_value=0.0, max_value=100.0, value=20.0)
feature9 = st.slider("FLOW (cm)", min_value=0.0, max_value=100.0, value=30.0)



if st.button("Predict"):
    input_data = pd.DataFrame({
        'Cement': [feature1],
        'Slag': [feature2],
        'Fly ash': [feature3],
        'Water': [feature4],
        'SP': [feature5],
        'Coarse Aggr.': [feature6],
        'Fine Aggr.': [feature7],
        'SLUMP(cm)': [feature8],
        'FLOW(cm)': [feature9]
    })
    prediction = loaded_model.predict(input_data)
    st.write(f"Predicted Compressive Strength (28-day)(Mpa): {prediction[0]:.2f}")