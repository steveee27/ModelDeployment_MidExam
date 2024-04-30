import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

model = joblib.load('xgb_classifier_model.pkl')
gender_encoder = joblib.load('gender_encoder.pkl')
hasCrCard_encoder = joblib.load('hasCrCard_encoder.pkl')
isActiveMember_encoder = joblib.load('isActiveMember_encoder.pkl')

def main():
    st.markdown("<h1 style='text-align: center;'>Churn Model Deployment</h1>", unsafe_allow_html=True)

    Surname = st.text_input("Surname: ")
    Age = st.number_input("Age: ", 0, 100)
    Gender = st.radio("Input Gender: ", ["Male","Female"])
    Geography = st.radio("Geography: ", ['France', 'Spain', 'Germany'])
    Tenure = st.selectbox("Tenure: ", list(range(1, 11)))
    Balance = st.number_input("Balance: ", 0, 10000000)
    NumOfProducts = st.selectbox("Number Of Products:", [1, 2, 3, 4])
    HasCrCard = st.radio("I Have a Credit Card: ", ["Yes","No"])
    IsActiveMember = st.radio("I am an Active Member : ", ["Yes","No"])
    EstimatedSalary = st.number_input("Estimated Salary: ", 0, 10000000)
    CreditScore = st.number_input("Credit Score: ", 0, 1000)

    geography_encoding = {'France': [0, 0, 1], 'Spain': [0, 1, 0], 'Germany': [1, 0, 0]}
    geography_encoded = geography_encoding[Geography]

    data = {'Surname': Surname, 'Age': int(Age), 'Gender': Gender, 
            'CreditScore':int(CreditScore),
            'Tenure': int(Tenure), 'Balance':int(Balance),
            'NumOfProducts': NumOfProducts, 'HasCrCard': HasCrCard,
            'IsActiveMember':IsActiveMember,'EstimatedSalary':int(EstimatedSalary),
            'Geography_France': geography_encoded[0],
            'Geography_Spain': geography_encoded[1],
            'Geography_Germany': geography_encoded[2]}
    
    df = pd.DataFrame([list(data.values())], columns = ['Surname', 
                                                        'Age',
                                                        'Gender',  
                                                        'CreditScore', 'Tenure',
                                                        'Balance', 
                                                        'NumOfProducts', 'HasCrCard' ,'IsActiveMember', 'EstimatedSalary',
                                                        'Geography_France', 'Geography_Spain', 'Geography_Germany'])
    
    scaler = StandardScaler()

    df[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']] = scaler.fit_transform(df[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']])

    df = df.replace(gender_encoder)
    df = df.replace(hasCrCard_encoder)
    df = df.replace(isActiveMember_encoder)

    if st.button('Make Prediction'):
        features = df.drop('Surname', axis=1)      
        result = makePrediction(features)
        prediction_text = "Churn" if result == 1 else "Not Churn"
        st.success(f"Mr./Mrs. {Surname} is {prediction_text}")

    st.markdown("<p style='text-align: center; font-size: small;'>Created by Steve Marcello Liem / 2602071410 / LA-09</p>", unsafe_allow_html=True)

def makePrediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
