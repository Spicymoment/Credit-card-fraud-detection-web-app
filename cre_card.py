# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:04:53 2023

@author: Spicy
"""
    
import streamlit as st
import pandas as pd
import pickle
#import base64

# Load the trained models
lr_model = pickle.load(open('C:/Users/User/Downloads/CC_FDWA/model_lr.pkl', 'rb'))
dt_model = pickle.load(open('C:/Users/User/Downloads/CC_FDWA/model_dtc.pkl', 'rb'))
rf_model = pickle.load(open('C:/Users/User/Downloads/CC_FDWA/model_rfc.pkl', 'rb'))
svc_model = pickle.load(open('C:/Users/User/Downloads/CC_FDWA/model_svc.pkl', 'rb'))
# Create a dictionary of models
models = {
    'Logistic Regression': lr_model,
    'Decision Tree Classifier': dt_model,
    'Random Forest Classifier': rf_model,
    'Support Vector Classifier': svc_model,
}

# Function to predict fraud using the selected model
def predict_fraud(model, data):
    prediction = model.predict(data)
    return prediction

# Streamlit web app
def main():
    st.title('Credit Card Fraud Detector')

    # Create a sidebar for model selection and user input
    st.sidebar.title('Model Selection')
    selected_model = st.sidebar.radio('Select a model for prediction', list(models.keys()))

    st.sidebar.title('User Input')

    # User input for feature values
    feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                     'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                     'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 
                     'Amount']


    values = {}
    for feature in feature_names:
        values[feature] = st.sidebar.number_input(f'Enter value for {feature}',  step=0.1 , placeholder= 0.00, key=feature)

    # Button to trigger prediction
    predict_button = st.button('Predict')

    # Perform prediction when the predict button is clicked
    if predict_button:
        # Perform prediction
        model = models[selected_model]
        input_data = pd.DataFrame([values])[feature_names]
        prediction = predict_fraud(model, input_data)

        # Display prediction result
        if prediction[0] == 0:
            st.markdown('<span style="color:green">Prediction: Not a Fraudulent Transaction</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:red">Prediction: Fraudulent Transaction</span>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()