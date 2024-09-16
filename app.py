import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load('./obesity_model.pkl')

# Set up the title and description of the app
st.title("Obesity Level Prediction App")
st.write("""
### Enter the details to predict your obesity level.
This model predicts obesity levels based on several input features.
""")

# Create input fields for each feature required by the model
height = st.number_input('Height (in meters)', value=1.75, format="%.2f")
weight = st.number_input('Weight (in kg)', value=70, format="%.1f")
family_history_with_overweight = st.selectbox('Family History of Overweight', [0, 1], help="0 for No, 1 for Yes")
SCC = st.selectbox('Self-Control over Calories (SCC)', [0, 1], help="0 for No, 1 for Yes")
MTRANS_Walking = st.selectbox('Main Transportation is Walking', [0, 1], help="0 for No, 1 for Yes")
FAVC_z = st.number_input('Consumption of High Caloric Food (z-score)', value=0.5, format="%.2f")
FCVC_minmax = st.number_input('Frequency of Eating Vegetables (min-max scaled)', value=0.7, format="%.2f")
NCP_z = st.number_input('Number of Meals (z-score)', value=0.8, format="%.2f")
CAEC_minmax = st.number_input('Eating between Meals (min-max scaled)', value=0.6, format="%.2f")
CH2O_minmax = st.number_input('Daily Water Intake (min-max scaled)', value=0.75, format="%.2f")
FAF_minmax = st.number_input('Physical Activity Frequency (min-max scaled)', value=0.65, format="%.2f")
TUE_z = st.number_input('Time Spent on Technology (z-score)', value=0.9, format="%.2f")
CALC_z = st.number_input('Alcohol Consumption (z-score)', value=0.3, format="%.2f")
Age_bin_minmax = st.number_input('Age (binned and min-max scaled)', value=0.55, format="%.2f")

# When the "Predict" button is clicked, make a prediction
if st.button('Predict Obesity Level'):
    # Create a feature array in the order expected by the model
    features = np.array([[height, weight, family_history_with_overweight, SCC, 
                          MTRANS_Walking, FAVC_z, FCVC_minmax, NCP_z, 
                          CAEC_minmax, CH2O_minmax, FAF_minmax, 
                          TUE_z, CALC_z, Age_bin_minmax]])
    
    # Make prediction using the loaded model
    prediction = model.predict(features)
    
    # Map prediction result to corresponding obesity level
    obesity_levels = [
        'Insufficient Weight', 
        'Normal Weight', 
        'Overweight Level I', 
        'Overweight Level II', 
        'Obesity Type I', 
        'Obesity Type II', 
        'Obesity Type III'
    ]
    
    # Display the prediction result
    st.write(f"Predicted Obesity Level: {obesity_levels[int(prediction[0])]}")



