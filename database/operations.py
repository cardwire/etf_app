# Store ETF data
store_etf_data(etf_df, server, database, username, password)


# Store Yahoo Finance data for SPY
store_yahoo_data(spy, server, database, username, password)







# Function to get the etf_df from the Azure SQL database
from sqlalchemy import create_engine
from dotenv import load_dotenv
from os import getenv
import urllib.parse


# Load the environment variables
load_dotenv()


# Define the connection string using the environment variables
server = os.getenv('AZURE_SQL_SERVER')
database = os.getenv('AZURE_SQL_DATABASE')
username = os.getenv('AZURE_SQL_USER')
password =  os.getenv('AZURE_SQL_PASSWORD')
driver = os.getenv('AZURE_SQL_DRIVER')

# Create the connection string
connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={urllib.parse.quote_plus(driver)}&TrustServerCertificate=yes'

# Create the engine
engine = create_engine(connection_string)

# Test the connection
try:
    with engine.connect() as connection:
        print("Connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")

def get_etf_data(server, database, username, password, table_name='etf_data'):
    connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+18+for+SQL+Server'
    engine = create_engine(connection_string)
    etf_df = pd.read_sql(f'SELECT * FROM {table_name}', con=engine)
    return etf_df



