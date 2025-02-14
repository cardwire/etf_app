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



#define the adaboost forecast function
#import numpy as np
#from sklearn.ensemble import AdaBoostRegressor
#from sklearn.tree import DecisionTreeRegressor

def ada_forecast(ticker, period):
    #create the history data in prophet style
    history = ticker.history(period='max')
    history = history.reset_index()
    history = history.rename(columns={'Date': 'ds', 'Close': 'y'})
    history["ds"] = history['ds'].dt.tz_localize(None)
    #create a prophet model                 
    X = np.array((history['ds'] - history['ds'].min()).dt.days).reshape(-1, 1)
    y = history['y']
    # create a future dataframe for the next 365 days
    # Create and fit the AdaBoost model
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=100, random_state=42)
    model.fit(X, y)

    # Create future dates
    future_dates = pd.date_range(start=history['ds'].max(), periods=period).to_frame(index=False, name='ds')
    future_X = np.array((future_dates['ds'] - history['ds'].min()).dt.days).reshape(-1, 1)

    # Make predictions
    forecast = model.predict(future_X)

    # plot the forecast using plotly
    fig = go.Figure()






def get_etf_data(server, database, username, password, table_name='etf_data'):
    connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+18+for+SQL+Server'
    engine = create_engine(connection_string)
    etf_df = pd.read_sql(f'SELECT * FROM {table_name}', con=engine)
    return etf_df



