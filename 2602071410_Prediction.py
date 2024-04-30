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
    st.title('Churn Model Deployment')

    Surname = st.text_input("Input Surname: ")
    Age = st.number_input("Input Age: ", 0, 100)
    Gender = st.radio("Input Gender: ", ["Male","Female"])
    Tenure = st.selectbox("Tenure: ", list(range(1, 11)))
    Balance = st.number_input("Balance: ", 0, 10000000)
    NumOfProducts = st.selectbox("Number Of Products:", [1, 2, 3, 4])
    HasCrCard = st.radio("I Have a Credit Card: ", ["Yes","No"])
    IsActiveMember = st.radio("I am an Active Member : ", ["Yes","No"])
    EstimatedSalary = st.number_input("Estimated Salary: ", 0, 10000000)
    CreditScore = st.number_input("Credit Score: ", 0, 1000)

    data = {'Surname': Surname, 'Age': int(Age), 'Gender': Gender, 
            'CreditScore':int(CreditScore),
            'Tenure': int(Tenure), 'Balance':int(Balance),
            'NumOfProducts': NumOfProducts, 'HasCrCard': HasCrCard,
            'IsActiveMember':IsActiveMember,'EstimatedSalary':int(EstimatedSalary)}
    
    df = pd.DataFrame([list(data.values())], columns = ['Surname', 
                                                        'Age',
                                                        'Gender',  
                                                        'CreditScore', 
                                                        'Tenure',
                                                        'Balance', 
                                                        'NumOfProducts', 
                                                        'HasCrCard',
                                                        'IsActiveMember',
                                                        'EstimatedSalary'])
            
    scaler = StandardScaler()

    df = df.replace(gender_encoder)
    df = df.replace(hasCrCard_encoder)
    df = df.replace(isActiveMember_encoder)

    df = scaler.fit_transform(df.drop('Surname', axis=1))

    if st.button('Make Prediction'):
        features = df      
        result = makePrediction(features)
        prediction_text = "Churn" if result == 1 else "Not Churn"
        st.success(f"Mr./Mrs. {Surname} is {prediction_text}")

def makePrediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()