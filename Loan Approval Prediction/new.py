#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
import streamlit as st

train = pd.read_csv(r"D:\ml_project\train.csv")

train = train.dropna()

train['TotalApplicantIncome'] = train['ApplicantIncome'] + train['CoapplicantIncome']

train = pd.get_dummies(train, columns=['Gender', 'Married', 'Loan_Status'], drop_first=True)
train = train.rename(columns={'Loan_Status_Y': 'Loan_Approved'})

train['Credit_History'] = train['Credit_History'].astype(int)

X = train[['Gender_Male', 'Married_Yes', 'TotalApplicantIncome', 'LoanAmount', 'Credit_History']]
y = train['Loan_Approved']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, shuffle=True)

forest = RandomForestClassifier(max_depth=4, random_state=10, n_estimators=100, min_samples_leaf=5)
model = forest.fit(x_train, y_train)

@st.cache_data
def prediction(Gender, Married, TotalApplicantIncome, LoanAmount, Credit_History):
    Gender = 0 if Gender == "Male" else 1
    Married = 0 if Married == "Unmarried" else 1
    Credit_History = 0 if Credit_History == "No Credit History" else 1
    LoanAmount = LoanAmount / 1000

    pred_inputs = model.predict([[Gender, Married, TotalApplicantIncome, LoanAmount, Credit_History]])

    if pred_inputs[0] == 0:
        pred = 'I am sorry, you have been rejected for the loan.'
    elif pred_inputs[0] == 1:
        pred = 'Congrats! You have been approved for the loan!'
    else:
        pred = 'Error'
    return pred

def main():
    html_temp = """ 
    <div style ="background-color:teal;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    Gender = st.selectbox('Gender', ("Male", "Female"))
    Married = st.selectbox('Marital Status', ("Unmarried", "Married"))
    TotalApplicantIncome = st.number_input("Total Monthly Income, (Include Coborrower if Applicable)")
    LoanAmount = st.number_input("Loan Amount (ex. 125000)")
    Credit_History = st.selectbox('Credit History', ("Has Credit History", "No Credit History"))
    result = ""

    if st.button("Predict"):
        result = prediction(Gender, Married, TotalApplicantIncome, LoanAmount, Credit_History)
        st.success('Final Decision: {}'.format(result))

if __name__ == '__main__':
    main()

