# Store ETF data
store_etf_data(etf_df, server, database, username, password)


# Store Yahoo Finance data for SPY
store_yahoo_data(spy, server, database, username, password)












def get_etf_data(server, database, username, password, table_name='etf_data'):
    connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+18+for+SQL+Server'
    engine = create_engine(connection_string)
    etf_df = pd.read_sql(f'SELECT * FROM {table_name}', con=engine)
    return etf_df



