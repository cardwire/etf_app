# Store ETF data
#store_etf_data(etf_df, server, database, username, password)


# Store Yahoo Finance data for SPY
#store_yahoo_data(spy, server, database, username, password)


# All functions required in this app



#define the prophet forecast function
def prophet_forecast(ticker, period):
    #create the history data in prophet style
    history = ticker.history(period='max')
    history = history.reset_index()
    history = history.rename(columns={'Date': 'ds', 'Close': 'y'})
    history['ds'] = pd.to_datetime(history['ds'])
    history["ds"] = history['ds'].dt.tz_localize(None)

    #create a prophet model                 
    model = Prophet()
    model.fit(history)
    # create a future dataframe for the next 365 days
    future = model.make_future_dataframe(periods=period)

    # make predictions
    forecast = model.predict(future)

    # plot the forecast using plotly
    fig = go.Figure()
        # Add the actual data
    fig.add_trace(go.Scatter(x=history['ds'], y=history['y'], mode='lines', name='Actual'))
    # Add the forecast data
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

    # Add the upper and lower bounds
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash')))

    # indicate the forecasted region with a vertical line at the last known date
    fig.add_vline(x=history['ds'].max(), line_width=2, line_dash="dash", line_color="black")

    #add slider to the plot to zoom in and out
    fig.update_layout(xaxis_rangeslider_visible=True)

    # Update layout
    fig.update_layout(title=f'Forecast for {ticker.ticker} for the next {period} days',
                      xaxis_title='Date',
                      yaxis_title='Price')

    st.plotly_chart(fig)


#define the adaboost forecast function
#import numpy as np
#from sklearn.ensemble import AdaBoostRegressor
#from sklearn.tree import DecisionTreeRegressor



def ada_forecast(ticker, period):
    history = ticker.history(period='max')
    history = history.reset_index()
    history = history.rename(columns={'Date': 'ds', 'Close': 'y'})
    history['ds'] = pd.to_datetime(history['ds'])
    history["ds"] = history['ds'].dt.tz_localize(None)
    
    X = np.array((history['ds'] - history['ds'].min()).dt.days).reshape(-1, 1)
    y = history['y'].values
    
    model = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=100)
    model.fit(X, y)
    
    future_dates = pd.date_range(start=history['ds'].max(), periods=period)
    future_X = np.array((future_dates - history['ds'].min()).days).reshape(-1, 1)
    forecast = model.predict(future_X)
    
    # Plotting code here...

    # plot the forecast using plotly
    fig = go.Figure()
        # Add the actual data
    fig.add_trace(go.Scatter(x=history['ds'], y=history['y'], mode='lines', name='Actual'))
    # Add the forecast data
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

    # Add the upper and lower bounds
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash')))

    # indicate the forecasted region with a vertical line at the last known date
    fig.add_vline(x=history['ds'].max(), line_width=2, line_dash="dash", line_color="black")

    #add slider to the plot to zoom in and out
    fig.update_layout(xaxis_rangeslider_visible=True)

    # Update layout
    fig.update_layout(title=f'Forecast for {ticker.ticker} for the next {period} days',
                      xaxis_title='Date',
                      yaxis_title='Price')

    st.plotly_chart(fig)





def get_etf_data(server, database, username, password, table_name='etf_data'):
    connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+18+for+SQL+Server'
    engine = create_engine(connection_string)
    etf_df = pd.read_sql(f'SELECT * FROM {table_name}', con=engine)
    return etf_df



