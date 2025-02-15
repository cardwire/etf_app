# Store ETF data
#store_etf_data(etf_df, server, database, username, password)
import pandas as pd
def add_esg(data, esg):
    esg = pd.read_csv("database/esg.csv")
    data = data.merge(esg, left_on="symbol", right_on="ticker", how="left")
    return data
    

# Store Yahoo Finance data for SPY
#store_yahoo_data(spy, server, database, username, password)
import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objs as go
import plotly.express as px
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from prophet import Prophet


# All functions required in this app



#define the prophet forecast function
from prophet import Prophet
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from prophet import Prophet
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def prophet_forecast(ticker, period):

    # Create the history data in prophet style
    history = ticker.history(period='max')  # Corrected syntax
    history = history.reset_index()
    history = history.rename(columns={'Date': 'ds', 'Close': 'y'})
    history['ds'] = pd.to_datetime(history['ds'])
    history["ds"] = history['ds'].dt.tz_localize(None)

    # Create a prophet model
    model = Prophet()
    model.fit(history)

    # Create a future dataframe for the next `period` days
    future = model.make_future_dataframe(periods=period)

    # Make predictions
    forecast = model.predict(future)

    # Plot the forecast using plotly
    fig = go.Figure()

    # Limit the time period to the same as the forecast
    past = datetime.today() - timedelta(days=period)
    future = datetime.today() + timedelta(days=period)

    # Filter historical data to the desired time window
    history_filtered = history[history['ds'] >= past]

    # Add the actual data
    fig.add_trace(go.Scatter(x=history_filtered['ds'], y=history_filtered['y'], mode='lines', name='Actual'))

    # Add the forecast data
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

    # Add the upper and lower bounds
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash')))

    # Indicate the forecasted region with a vertical line at the last known date
    fig.add_vline(x=history['ds'].max(), line_width=2, line_dash="dash", line_color="black")

    # Update layout to limit the x-axis range and set titles
    fig.update_layout(
        xaxis_range=[past, future],  # Limit the x-axis to the desired time window
        title=f'Forecast for {ticker.ticker} for the next {period} days',
        xaxis_title='Date',
        yaxis_title='Price'
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

###################################################################################

def ar_forecast:
    # Create the history data
    history = ticker.history(period='max')
    history = history.reset_index()
    history = history.rename(columns={'Date': 'ds', 'Close': 'y'})
    history['ds'] = pd.to_datetime(history['ds'])
    history["ds"] = history['ds'].dt.tz_localize(None)

    # Prepare data for AR model
    y = history['y'].values  # Historical closing prices as a NumPy array
    dates = history['ds'].values  # Historical dates as a NumPy array

    # Fit the AR model
    model = AutoReg(y, lags=30)  # Adjust lags as needed
    model_fit = model.fit()

    # Make predictions
    forecast = model_fit.predict(start=len(y), end=len(y) + period - 1)

    # Create future dates
    future_dates = pd.date_range(start=history['ds'].max() + timedelta(days=1), periods=period)

    # Plot the forecast using plotly
    fig = go.Figure()

    # Limit the time period to the same as the forecast
    past = datetime.today() - timedelta(days=period)
    future = datetime.today() + timedelta(days=period)

    # Filter historical data to the desired time window
    history_filtered = history[history['ds'] >= past]

    # Add the actual data
    fig.add_trace(go.Scatter(x=history_filtered['ds'], y=history_filtered['y'], mode='lines', name='Actual'))

    # Add the forecast data
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Forecast'))

    # Update layout to limit the x-axis range and set titles
    fig.update_layout(
        xaxis_range=[past, future],  # Limit the x-axis to the desired time window
        title=f'AR Forecast for {ticker.ticker} for the next {period} days',
        xaxis_title='Date',
        yaxis_title='Price'
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

import plotly.express as px

fig_acf = px.line(x=range(len(y)), y=plot_acf(y, lags=50).values, title='Autocorrelation')
fig_acf.show() # Plot autocorrelation for the first 50 lags
plt.show()



#########################################################################################################

from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(ticker, period):
    # Create the history data in a suitable format
    history = ticker.history(period='max')
    history = history.reset_index()
    history = history.rename(columns={'Date': 'ds', 'Close': 'y'})
    history['ds'] = pd.to_datetime(history['ds'])
    history["ds"] = history['ds'].dt.tz_localize(None)

    # Prepare the data for ARIMA model
    y = history['y']

    # Create and fit the ARIMA model
    model = ARIMA(y, order=(5, 1, 0))
    model_fit = model.fit()

    # Create future dates
    future_dates = pd.date_range(start=history['ds'].max(), periods=period).to_frame(index=False, name='ds')

    # Make predictions
    forecast = model_fit.forecast(steps=period)

    # Plot the forecast using plotly
    fig = go.Figure()

    # Add the actual data
    fig.add_trace(go.Scatter(x=history['ds'], y=history['y'], mode='lines', name='Actual'))

    # Add the forecast data
    fig.add_trace(go.Scatter(x=future_dates['ds'], y=forecast, mode='lines', name='Forecast'))

    try:
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash')))
    except Exception as e:
        st.error(f"Error adding traces: {e}")

    
    # Update layout
    fig.update_layout(title=f'Forecast for {ticker.ticker} for the next {period} days using ARIMA',
                      xaxis_title='Date',
                      yaxis_title='Price')

    # Indicate the forecasted region with a vertical line at the last known date
    fig.add_vline(x=history['ds'].max(), line_width=2, line_dash="dash", line_color="black")

    # Add slider to the plot to zoom in and out
    fig.update_layout(xaxis_rangeslider_visible=True)

    fig.show()

    st.plotly_chart(fig)





########################################################################################################################





from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima_forecast(ticker, period):
    # Create the history data in a suitable format
    history = ticker.history(period='max')
    history = history.reset_index()
    history = history.rename(columns={'Date': 'ds', 'Close': 'y'})
    history['ds'] = pd.to_datetime(history['ds'])
    history["ds"] = history['ds'].dt.tz_localize(None)

    # Prepare the data for SARIMA model
    y = history['y']

    # Create and fit the SARIMA model
    model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)

    # Create future dates
    future_dates = pd.date_range(start=history['ds'].max(), periods=period).to_frame(index=False, name='ds')

    # Make predictions
    forecast = model_fit.forecast(steps=period)

    # Plot the forecast using plotly
    fig = go.Figure()

    # Add the actual data
    fig.add_trace(go.Scatter(x=history['ds'], y=history['y'], mode='lines', name='Actual'))

    # Add the forecast data
    fig.add_trace(go.Scatter(x=future_dates['ds'], y=forecast, mode='lines', name='Forecast'))

    try:
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash')))
    except Exception as e:
        st.error(f"Error adding traces: {e}")
    
    # Update layout
    fig.update_layout(title=f'Forecast for {ticker.ticker} for the next {period} days using SARIMA',
                      xaxis_title='Date',
                      yaxis_title='Price')

    # Indicate the forecasted region with a vertical line at the last known date
    fig.add_vline(x=history['ds'].max(), line_width=2, line_dash="dash", line_color="black")

    # Add slider to the plot to zoom in and out
    fig.update_layout(xaxis_rangeslider_visible=True)

    fig.show()

    st.plotly_chart(fig)


###################################################################

from statsmodels.tsa.holtwinters import ExponentialSmoothing

def es_forecast(ticker, period):
    # Create the history data in a suitable format
    history = ticker.history(period='max')
    history = history.reset_index()
    history = history.rename(columns={'Date': 'ds', 'Close': 'y'})
    history['ds'] = pd.to_datetime(history['ds'])
    history["ds"] = history['ds'].dt.tz_localize(None)

    # Prepare the data for Exponential Smoothing model
    y = history['y']

    # Create and fit the Exponential Smoothing model
    model = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=12)
    model_fit = model.fit()

    # Create future dates
    future_dates = pd.date_range(start=history['ds'].max(), periods=period).to_frame(index=False, name='ds')

    # Make predictions
    forecast = model_fit.forecast(steps=period)

    # Plot the forecast using plotly
    fig = go.Figure()

    # Add the actual data
    fig.add_trace(go.Scatter(x=history['ds'], y=history['y'], mode='lines', name='Actual'))

    # Add the forecast data
    fig.add_trace(go.Scatter(x=future_dates['ds'], y=forecast, mode='lines', name='Forecast'))


    try:
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash')))
    except Exception as e:
        st.error(f"Error adding traces: {e}")

    
    # Update layout
    fig.update_layout(title=f'Forecast for {ticker.ticker} for the next {period} days using Exponential Smoothing',
                      xaxis_title='Date',
                      yaxis_title='Price')

    # Indicate the forecasted region with a vertical line at the last known date
    fig.add_vline(x=history['ds'].max(), line_width=2, line_dash="dash", line_color="black")

    # Add slider to the plot to zoom in and out
    fig.update_layout(xaxis_rangeslider_visible=True)

    fig.show()

    st.plotly_chart(fig)


################################################################################################


from xgboost import XGBRegressor

def xgb_forecast(ticker, period):
    # Create the history data in a suitable format
    history = ticker.history(period='max')
    history = history.reset_index()
    history = history.rename(columns={'Date': 'ds', 'Close': 'y'})
    history['ds'] = pd.to_datetime(history['ds'])
    history["ds"] = history['ds'].dt.tz_localize(None)

    # Prepare the data for XGBoost model
    X = np.array((history['ds'] - history['ds'].min()).dt.days).reshape(-1, 1)
    y = history['y']

    # Create and fit the XGBoost model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X, y)

    # Create future dates
    future_dates = pd.date_range(start=history['ds'].max(), periods=period).to_frame(index=False, name='ds')
    future_X = np.array((future_dates['ds'] - history['ds'].min()).dt.days).reshape(-1, 1)

    # Make predictions
    forecast = model.predict(future_X)

    # Plot the forecast using plotly
    fig = go.Figure()

    # Add the actual data
    fig.add_trace(go.Scatter(x=history['ds'], y=history['y'], mode='lines', name='Actual'))

    # Add the forecast data
    fig.add_trace(go.Scatter(x=future_dates['ds'], y=forecast, mode='lines', name='Forecast'))

    try:
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash')))
    except Exception as e:
        st.error(f"Error adding traces: {e}")

    
    # Update layout
    fig.update_layout(title=f'Forecast for {ticker.ticker} for the next {period} days using XGBoost',
                      xaxis_title='Date',
                      yaxis_title='Price')

    # Indicate the forecasted region with a vertical line at the last known date
    fig.add_vline(x=history['ds'].max(), line_width=2, line_dash="dash", line_color="black")

    # Add slider to the plot to zoom in and out
    fig.update_layout(xaxis_rangeslider_visible=True)

    fig.show()

    st.plotly_chart(fig)


###############################################################################################

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def lstm_forecast(ticker, period):
    # Create the history data in a suitable format
    history = ticker.history(period='max')
    history = history.reset_index()
    history = history.rename(columns={'Date': 'ds', 'Close': 'y'})
    history['ds'] = pd.to_datetime(history['ds'])
    history["ds"] = history['ds'].dt.tz_localize(None)

    # Prepare the data for LSTM model
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(history['y'].values.reshape(-1, 1))

    # Create the training data
    train_data = []
    for i in range(60, len(scaled_data)):
        train_data.append(scaled_data[i-60:i, 0])
    
    train_data = np.array(train_data)
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Create and fit the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=25, batch_size=32)

    # Create future dates
    future_dates = pd.date_range(start=history['ds'].max(), periods=period).to_frame(index=False, name='ds')
    future_X = np.array((future_dates['ds'] - history['ds'].min()).dt.days).reshape(-1, 1)

    # Make predictions
    inputs = history['y'][len(history) - len(future_dates) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, len(inputs)):
        X_test.append(inputs[i-60:i, 0])
    
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    forecast = model.predict(X_test)
    forecast = scaler.inverse_transform(forecast)

    # Plot the forecast using plotly
    fig = go.Figure()

    # Add the actual data
    fig.add_trace(go.Scatter(x=history['ds'], y=history['y'], mode='lines', name='Actual'))


    try:
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash')))
    except Exception as e:
        st.error(f"Error adding traces: {e}")

    
    # Add the forecast data
    fig.add_trace(go.Scatter(x=future_dates['ds'], y=forecast.flatten(), mode='lines', name='Forecast'))

    # Update layout
    fig.update_layout(title=f'Forecast for {ticker.ticker} for the next {period} days using LSTM',
                      xaxis_title='Date',
                      yaxis_title='Price')

    # Indicate the forecasted region with a vertical line at the last known date
    fig.add_vline(x=history['ds'].max(), line_width=2, line_dash="dash", line_color="black")

    # Add slider to the plot to zoom in and out
    fig.update_layout(xaxis_rangeslider_visible=True)

    fig.show()

    st.plotly_chart(fig)

########################################################################################################################

'''

import gluonts
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions

def deepar_forecast(ticker, period):
    # Create the history data in a suitable format
    history = ticker.history(period='max')
    history = history.reset_index()
    history = history.rename(columns={'Date': 'ds', 'Close': 'y'})
    history['ds'] = pd.to_datetime(history['ds'])
    history["ds"] = history['ds'].dt.tz_localize(None)

    # Prepare the data for DeepAR model
    training_data = ListDataset(
        [{"start": history['ds'].iloc[0], "target": history['y'].values}],
        freq="D"
    )

    # Create and train the DeepAR model
    estimator = DeepAREstimator(
        freq="D",
        prediction_length=period,
        trainer=Trainer(epochs=10)  # Reduce epochs for faster testing
    )
    predictor = estimator.train(training_data)

    # Make predictions
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=training_data,
        predictor=predictor,
        num_samples=100
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    # Extract forecast values
    forecast = forecasts[0].mean

    # Create future dates
    future_dates = pd.date_range(
        start=history['ds'].max() + pd.Timedelta(days=1),
        periods=period
    ).to_frame(index=False, name='ds')

    # Plot the forecast using plotly
    fig = go.Figure()

    # Add the actual data
    fig.add_trace(go.Scatter(
        x=history['ds'],
        y=history['y'],
        mode='lines',
        name='Actual'
    ))

    # Add the forecast data
    fig.add_trace(go.Scatter(
        x=future_dates['ds'],
        y=forecast,
        mode='lines',
        name='Forecast'
    ))

    try:
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash')))
    except Exception as e:
        st.error(f"Error adding traces: {e}")

    # Update layout
    fig.update_layout(
        title=f'Forecast for {ticker.ticker} for the next {period} days using DeepAR',
        xaxis_title='Date',
        yaxis_title='Price'
    )

    # Indicate the forecasted region with a vertical line at the last known date
    fig.add_vline(
        x=history['ds'].max(),
        line_width=2,
        line_dash="dash",
        line_color="black"
    )

    # Add slider to the plot to zoom in and out
    fig.update_layout(xaxis_rangeslider_visible=True)

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    st.plotly_chart(fig)


################################################################################################################################


import gluonts
from gluonts.dataset.common import ListDataset
from gluonts.model.n_beats import NBEATSEstimator
from gluonts.mx.trainer import Trainer
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions

def nbeats_forecast(ticker, period):
    # Create the history data in a suitable format
    history = ticker.history(period='max')
    history = history.reset_index()
    history = history.rename(columns={'Date': 'ds', 'Close': 'y'})
    history['ds'] = pd.to_datetime(history['ds'])
    history["ds"] = history['ds'].dt.tz_localize(None)

    # Prepare the data for N-BEATS model
    training_data = ListDataset(
        [{"start": history['ds'].iloc[0], "target": history['y'].values}],
        freq="D"
    )

    # Create and train the N-BEATS model
    estimator = NBEATSEstimator(freq="D", prediction_length=period, trainer=Trainer(epochs=25))
    predictor = estimator.train(training_data)

    # Make predictions
    forecast_it, ts_it = make_evaluation_predictions(training_data, predictor=predictor, num_samples=100)
    forecasts = list(forecast_it)
    tss = list(ts_it)

    # Extract forecast values
    forecast = forecasts[0].mean

    # Create future dates
    future_dates = pd.date_range(start=history['ds'].max() + pd.Timedelta(days=1), periods=period).to_frame(index=False, name='ds')

    # Plot the forecast using plotly
    fig = go.Figure()

    # Add the actual data
    fig.add_trace(go.Scatter(x=history['ds'], y=history['y'], mode='lines', name='Actual'))

    # Add the forecast data
    fig.add_trace(go.Scatter(x=future_dates['ds'], y=forecast, mode='lines', name='Forecast'))


    try:
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash')))
    except Exception as e:
        st.error(f"Error adding traces: {e}")


    # Update layout
    fig.update_layout(title=f'Forecast for {ticker.ticker} for the next {period} days using N-BEATS',
                      xaxis_title='Date',
                      yaxis_title='Price')

    # Indicate the forecasted region with a vertical line at the last known date
    fig.add_vline(x=history['ds'].max(), line_width=2, line_dash="dash", line_color="black")

    # Add slider to the plot to zoom in and out
    fig.update_layout(xaxis_rangeslider_visible=True)

    fig.show()




    st.plotly_chart(fig)

############################################################################################################################

from gluonts.dataset.common import ListDataset
from gluonts.model.tft import TemporalFusionTransformerEstimator
from gluonts.mx.trainer import Trainer
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions

def tft_forecast(ticker, period):
    # Create the history data in a suitable format
    history = ticker.history(period='max')
    history = history.reset_index()
    history = history.rename(columns={'Date': 'ds', 'Close': 'y'})
    history['ds'] = pd.to_datetime(history['ds'])
    history["ds"] = history['ds'].dt.tz_localize(None)

    # Prepare the data for TFT model
    training_data = ListDataset(
        [{"start": history['ds'].iloc[0], "target": history['y'].values}],
        freq="D"
    )

    # Create and train the TFT model
    estimator = TemporalFusionTransformerEstimator(freq="D", prediction_length=period, trainer=Trainer(epochs=25))
    predictor = estimator.train(training_data)

    # Make predictions
    forecast_it, ts_it = make_evaluation_predictions(training_data, predictor=predictor, num_samples=100)
    forecasts = list(forecast_it)
    tss = list(ts_it)

    # Extract forecast values
    forecast = forecasts[0].mean

    # Create future dates
    future_dates = pd.date_range(start=history['ds'].max() + pd.Timedelta(days=1), periods=period).to_frame(index=False, name='ds')

    # Plot the forecast using plotly
    fig = go.Figure()

    # Add the actual data
    fig.add_trace(go.Scatter(x=history['ds'], y=history['y'], mode='lines', name='Actual'))

    try:
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash')))
    except Exception as e:
        st.error(f"Error adding traces: {e}")

    # Add the forecast data
    fig.add_trace(go.Scatter(x=future_dates['ds'], y=forecast, mode='lines', name='Forecast'))

    # Update layout
    fig.update_layout(title=f'Forecast for {ticker.ticker} for the next {period} days using TFT',
                      xaxis_title='Date',
                      yaxis_title='Price')

    # Indicate the forecasted region with a vertical line at the last known date
    fig.add_vline(x=history['ds'].max(), line_width=2, line_dash="dash", line_color="black")

    # Add slider to the plot to zoom in and out
    fig.update_layout(xaxis_rangeslider_visible=True)

    fig.show()

    st.plotly_chart(fig)
'''

####################################################################################################################################
def get_etf_data(server, database, username, password, table_name='etf_data'):
    connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+18+for+SQL+Server'
    engine = create_engine(connection_string)
    etf_df = pd.read_sql(f'SELECT * FROM {table_name}', con=engine)
    return etf_df



