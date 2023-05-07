import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.mixture import GaussianMixture

# Define the app
st.set_page_config(page_title="Netflix Stock Price Prediction",
                   page_icon=":chart_with_upwards_trend:",
                   layout="wide")

@st.cache_data()
def load_data():
    df = pd.read_csv('NFLX.csv', parse_dates=['Date'], index_col='Date')
    return df



df = load_data()


# Define the ARIMA model
def fit_arima_model(order=(1, 1, 1)):
    model = ARIMA(df['Close'], order=order)
    model_fit = model.fit()
    return model_fit


# Define the GMM model
def fit_gmm_model():
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(df['Close'].values.reshape(-1, 1))
    return gmm


st.title('Netflix Stock Price Prediction')
st.markdown('''
This app predicts the next 30 days of Netflix stock prices using the ARIMA model.
Use the sliders to adjust the model parameters.
''')

# Show the data
if st.checkbox('Show Data'):
    st.write(df)

# Allow the user to manipulate the model
st.sidebar.subheader('Model Parameters')
order_1 = st.sidebar.slider('Order 1', min_value=1, max_value=5, value=1, step=1)
order_2 = st.sidebar.slider('Order 2', min_value=1, max_value=2, value=1, step=1)
order_3 = st.sidebar.slider('Order 3', min_value=1, max_value=2, value=1, step=1)
order = (order_1, order_2, order_3)
model_fit = fit_arima_model(order)


# Show the visualization of the Close price
fig1, ax1 = plt.subplots()
ax1.plot(df.index, df['Close'])
ax1.set_title('Netflix Stock Close Price')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price ($)')
st.pyplot(fig1)

# Make the prediction
forecast = model_fit.forecast(steps=30)
forecast_dates = pd.date_range(start=df.index[-1], periods=30, freq='B')

# Show the prediction
st.subheader('Next 30 Days Price Prediction')
pred_df = pd.DataFrame({'Date': forecast_dates, 'Close': forecast})
if st.checkbox('Show Prediction Data'):
    st.write(pred_df)

# Allow the user to select his own visualizations view
visualization_options = ['Actual vs Prediction', 'Prediction Distribution']
visualization_selected = st.selectbox('Select Visualization', visualization_options)

# Show the visualization of the prediction
if visualization_selected == 'Actual vs Prediction':
    fig2, ax2 = plt.subplots()
    ax2.plot(df.index, df['Close'], label='Actual')
    ax2.plot(forecast_dates, forecast, label='Prediction')
    ax2.set_title('Netflix Stock Close Price Prediction')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    st.pyplot(fig2)
elif visualization_selected == 'Prediction Distribution':
    gmm = fit_gmm_model()
    fig3, ax3 = plt.subplots()
    ax3.hist(df['Close'], bins=30, alpha=0.5, label='Actual', density=True)
    ax3.hist(forecast, bins=30, alpha=0.5, label='Prediction', density=True)
    x = np.linspace(df['Close'].min(), df['Close'].max(), 100)
    y = np.exp(gmm.score_samples(x.reshape(-1, 1)))
        # Plot the Gaussian Mixture Model density
    ax3.plot(x, y, '-r', label='Density')
    ax3.set_title('Netflix Stock Close Price Prediction Distribution')
    ax3.set_xlabel('Price ($)')
    ax3.set_ylabel('Density')
    ax3.legend()
    st.pyplot(fig3)

