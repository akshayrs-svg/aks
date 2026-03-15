import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from prophet import Prophet
from sklearn.metrics import root_mean_squared_error
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# Five years of historical data
start_date = date.today() - timedelta(days=5*365)
today = date.today()

st.title("Silver Price Prediction App")
st.markdown("Extracts the last 5 years of Silver price using Yahoo Finance & predicts for next one with low RMSE.")

@st.cache_data
def load_data():
    df = yf.download("SI=F", start=start_date, end=today)
    df.reset_index(inplace=True)
    return df

data = load_data()

# yf.download can return MultiIndex columns in recent versions
if isinstance(data.columns, pd.MultiIndex):
    # Flatten columns by just keeping the first level to avoid confusion
    data.columns = [col[0] for col in data.columns]

st.subheader('Raw Data (Last 5 Years)')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    # Safely extract close prices
    close_price = data['Close']
    if isinstance(close_price, pd.DataFrame):
        close_price = close_price.iloc[:, 0]
        
    fig.add_trace(go.Scatter(x=data['Date'], y=close_price, name='Silver Close Price'))
    fig.layout.update(title_text='Historical Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Data Preparation for Prophet Model
df_train = data[['Date', 'Close']]

if isinstance(df_train['Close'], pd.DataFrame):
    df_train['Close'] = df_train['Close'].iloc[:, 0]

df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
df_train = df_train.dropna()

st.subheader("Training Prophet Model")
with st.spinner('Training model...'):
    m = Prophet(daily_seasonality=True)
    m.fit(df_train)

# Calculate RMSE on training data
train_pred = m.predict(df_train)
rmse = root_mean_squared_error(df_train['y'], train_pred['yhat'])

st.write(f"**Model Training RMSE:** {rmse:.2f}")
st.markdown("*A lower RMSE value means a better fit to the historical data.*")

# Predict next 1 year (365 days)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

st.subheader('Forecasted Data (Next 1 Year)')
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

st.write('### Forecast Plot')
fig1 = plot_plotly(m, forecast)
fig1.layout.update(title_text='Forecast For Next 1 Year', xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)
