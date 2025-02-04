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


# Functions to create different Database Tables


# Function to store ETF data in Azure SQL database
def store_etf_data(df, server, database, username, password, table_name='etf_data'):
	connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+18+for+SQL+Server'
	engine = create_engine(connection_string)
	df.to_sql(table_name, con=engine, if_exists='replace', index=False)
	print("ETF data stored in database.")


# Function to store sector weightings in the database
def store_sector_weightings(sector_weightings, server, database, username, password, table_name='sector_weightings'):
    connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+18+for+SQL+Server'
    engine = create_engine(connection_string)
    
    # Convert the sector weightings dictionary to a DataFrame
    sector_weightings_df = pd.DataFrame.from_dict(sector_weightings, orient='index').reset_index()
    sector_weightings_df.columns = ['symbol'] + list(sector_weightings_df.columns[1:])
    
    # Store the DataFrame in the database
    sector_weightings_df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    print("Sector weightings stored in database.")












