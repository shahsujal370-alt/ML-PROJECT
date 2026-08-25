import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title='Loan Pricing Fairness Monitor', layout='wide')
st.title('Loan Pricing Fairness & Anomaly Detection')
st.caption('Risk-adjusted monitoring dashboard — anomalies are investigation flags, not proof of unfair treatment.')

@st.cache_resource
def load_models():
    return joblib.load('models/loan_pricing_model.joblib'), joblib.load('models/profile_anomaly_model.joblib')

@st.cache_data
def load_results():
    return pd.read_csv('outputs/test_predictions_and_anomalies.csv')

model, iso = load_models(); df = load_results()

c1,c2,c3,c4 = st.columns(4)
c1.metric('Loans in analysis', f'{len(df):,}')
c2.metric('Pricing anomalies', f"{df['pricing_anomaly'].sum():,}")
c3.metric('Peer anomalies', f"{df['peer_anomaly'].sum():,}")
c4.metric('High-priority overlap', f"{df['high_priority_anomaly'].sum():,}")

st.subheader('Pricing anomaly review')
threshold = float(df['abs_residual'].quantile(.99))
show = df[df['pricing_anomaly']].copy().sort_values('abs_residual', ascending=False)
cols = ['customer_age','customer_income','loan_amnt','loan_grade','term_years','actual_rate','predicted_rate','residual','peer_gap','profile_anomaly','high_priority_anomaly']
st.dataframe(show[cols].head(100), use_container_width=True)

st.subheader('Segment monitoring')
segment = st.selectbox('Segment', ['age_group','home_ownership','loan_grade','loan_intent'])
summary = df.groupby(segment, observed=True).agg(
    loans=('actual_rate','size'),
    mean_actual_rate=('actual_rate','mean'),
    mean_expected_rate=('predicted_rate','mean'),
    mean_residual=('residual','mean'),
    anomaly_rate=('pricing_anomaly','mean')
).reset_index()
st.dataframe(summary, use_container_width=True)

st.subheader('Single-loan pricing estimate')
with st.form('estimate'):
    age=st.number_input('Customer age',18,100,30)
    income=st.number_input('Customer income',0.0,1000000.0,50000.0)
    home=st.selectbox('Home ownership', sorted(df['home_ownership'].dropna().unique()))
    emp=st.number_input('Employment duration',0.0,60.0,5.0)
    intent=st.selectbox('Loan intent', sorted(df['loan_intent'].dropna().unique()))
    grade=st.selectbox('Loan grade', sorted(df['loan_grade'].dropna().unique()))
    amount=st.number_input('Loan amount',500.0,100000.0,10000.0)
    rate=None
    term=st.number_input('Term (years)',1,15,5)
    default=st.selectbox('Historical default', ['N','Y','Unknown'])
    hist=st.number_input('Credit history length',0,50,5)
    submitted=st.form_submit_button('Estimate expected interest rate')

if submitted:
    row=pd.DataFrame([{'customer_age':age,'customer_income':income,'home_ownership':home,'employment_duration':emp,'loan_intent':intent,'loan_grade':grade,'loan_amnt':amount,'term_years':term,'historical_default':default,'cred_hist_length':hist}])
    expected=float(model.predict(row)[0])
    st.success(f'Estimated expected interest rate: {expected:.2f}%')
    st.info('This is a model benchmark, not a lending decision or a legally defined fair rate.')
