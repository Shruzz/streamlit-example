from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st


import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import get_prediction, ordinal_encoder

model = joblib.load(r'MentalHealth.ipynb')

st.set_page_config(page_title="Students Mental Health Prediction App",
                   , layout="wide")

gender=['Male','Female']
Age=['18.','19.','20.','21.','22.','23.','24,']
course=['Engineering', 'religious', 'Arts', 'Law', 'Mathemathics',
       'Home Science', 'CA', 'Human Resources', 'Music', 'Psychology',
       'KENMS', 'Accounting ', 'ENM', 'Marine science', 'KOE',
       'Banking Studies', 'Business Administration', 'Usuluddin ',
       'TAASL', 'ALA', 'Biomedical science', 'Koe', 'BENL', 'CTS',
       'Econs', 'MHSC', 'Malcom', 'Kop', 'Human Sciences ',
       'Biotechnology', 'Communication ', 'Diploma Nursing',
       'Pendidikan Islam ', 'Radiography', 'Fiqh', 'DIPLOMA TESL',
       'Nursing ']
year=['1','2','3','4']
cgpa=['3.00 - 3.49', '3.50 - 4.00', '2.50 - 2.99', '2.00 - 2.49',
       '0 - 1.99']
martial=['Yes','No']
depression=['Yes','No']
anxiety=['Yes','No']
panic=['Yes','No']
treatment=['Yes','No']

features = ['Gender','Age','Course','Year of Study','CGPA','Martial Status','Have depression?','Have any anxiety?','Had any panic attacks before?']


st.markdown("<h1 style='text-align: center;'color:red;'>Mental Health Predictor </h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following :")
        
        gender = st.selectbox("Select your gender: ", option=gender)
        age = st.selectbox("Select your age: ", options=age)
        course = st.selectbox("Select your current pursuing course: ", options=course)
        year = st.selectbox("Select your year of study: ", options=year)
        cgpa = st.selectbox("Select CGPA score: ", options=cgpa)
        martial = st.selectbox("Select your martial status: ", options=martial)
        depression = st.selectbox("Do you have depression ", options=depression)
        anxiety = st.selectbox("Have you experienced any anxiety lately? ", options=anxiety)
        panic = st.selectbox("Had any panic attacks recently ? ", options=panic)
        
        
        
        submit = st.form_submit_button("Predict")


    if submit:
        gender = ordinal_encoder(gender ,gender)
        age = ordinal_encoder(age,age)
        course = ordinal_encoder(course,course)
        year =  ordinal_encoder(year,year)
        cgpa =  ordinal_encoder(cgpa,cgpa)
        martial = ordinal_encoder(martial,martial) 
        depression = ordinal_encoder(depression,depression)
        anxiety=ordinal_encoder(anxiety,anxiety)
        panic=ordinal_encoder(panic,panic)

        data = np.array( ['Gender','Age','Course','Year of Study','CGPA','Martial Status','Have depression?','Have any anxiety?','Had any panic attacks before?']).reshape(1,-1)

        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted severity is:  {pred[0]}")

if __name__ == '__main__':
    main()


