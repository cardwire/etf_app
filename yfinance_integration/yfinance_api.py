# Function to get sector weightings for a list of symbols
def get_sector_weightings(symbols):
    sector_weightings = {}
    for symbol in tqdm(symbols, desc="Fetching sector weightings"):
        try:
            sector_weightings[symbol] = yf.Ticker(symbol).get_funds_data().sector_weightings
            time.sleep(1)  # Sleep to reduce traffic
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    return sector_weightings
