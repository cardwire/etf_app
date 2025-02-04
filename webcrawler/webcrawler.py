import pypyodbc as odbc

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import tqdm
from time import sleep

etf_data = pd.DataFrame(columns=['symbol', 'fund_name', 'asset_class', 'assets'])


def get_etf_data():
    base_url = 'https://stockanalysis.com/etf/'
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    driver.get(base_url)

    etf_data = []
    unique_symbols = set()  # Track unique ETF symbols

    for page in tqdm.tqdm(range(1, 9)):  # Loop through 8 pages (adjust as needed)
        try:
            # Wait for the table to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'tbody'))
            )

            # Extract data from the table
            rows = driver.find_elements(By.CSS_SELECTOR, 'tbody tr')
            for row in rows:
                columns = row.find_elements(By.TAG_NAME, 'td')
                if len(columns) >= 4:  # Ensure the row has enough columns
                    symbol = columns[0].text.strip()

                    # Skip if the symbol is already scraped
                    if symbol in unique_symbols:
                        continue

                    fund_name = columns[1].text.strip()
                    asset_class = columns[2].text.strip()
                    assets = columns[3].text.strip()

                    etf_data.append([symbol, fund_name, asset_class, assets])
                    unique_symbols.add(symbol)  # Add the symbol to the set

                    # Add a pause after every 500 ETFs
                    if len(etf_data) % 500 == 0:
                        print(f"Scraped {len(etf_data)} ETFs. Pausing for 5 seconds...")
                        sleep(5)  # Adjust the pause duration as needed

            # Wait for any overlay to disappear
            try:
                WebDriverWait(driver, 5).until(
                    EC.invisibility_of_element_located((By.CSS_SELECTOR, '.fixed.left-0.top-0.z-\\[99\\]'))
                )
            except:
                pass  # If no overlay is found, continue

            # Scroll the "Next" button into view and click it
            next_button = driver.find_element(By.XPATH, '//span[contains(text(), "Next")]/parent::button')
            driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
            driver.execute_script("arguments[0].click();", next_button)

            # Wait for the next page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'tbody'))
            )
        except Exception as e:
            print(etf_df)
            print(f"Error on page {page}: {e}")
            break
    return pd.DataFrame(etf_data, columns=['symbol', 'fund_name', 'asset_class', 'assets'])

# Run the function and print the results
etf_df = get_etf_data()
