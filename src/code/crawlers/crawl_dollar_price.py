import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd

BASE_URL = 'http://www2.bmf.com.br/pages/portal/bmfbovespa/lumis/lum-retroativo-por-dia-enUS.asp'

def match_trading_rate_caption(table):
  return table.caption.text == 'Trading Rate'


def get_target_table_for_date(target_date):
  data = {
    'dData1': target_date
  }

  response = requests.post(BASE_URL, data=data)
  soup = BeautifulSoup(response.content, 'html.parser')

  tables = soup.find_all('table')
  target_tables = list(filter(match_trading_rate_caption, tables))

  return target_tables[0]

def append_row_data_to_df_dictionary(base_df_dictionary, target_table):
  row = target_table.find_all('tr')[-1]
  columns = row.find_all('td')

  if columns[0].text != 'USD':
    return

  base_df_dictionary['currency'].append(columns[0].text)
  base_df_dictionary['trading_date'].append(columns[1].text)
  base_df_dictionary['settlement_date'].append(columns[2].text)

  base_df_dictionary['minimum'].append(columns[3].text)
  base_df_dictionary['maximum'].append(columns[4].text)
  base_df_dictionary['last'].append(columns[5].text)

def main():

  base_df_dictionary = {
    'currency': [],
    'trading_date': [],
    'settlement_date': [],
    'minimum': [],
    'maximum': [],
    'last': [],
  }

  for year in tqdm(range(2011, 2021)):
    for month in range(1, 13):
      for day in range(1, 32):
        try:
          target_date = f'{month}/{day}/{year}'
          target_table = get_target_table_for_date(target_date)
          append_row_data_to_df_dictionary(base_df_dictionary, target_table)
        except:
          continue

    df = pd.DataFrame(base_df_dictionary)
    df.to_csv(f'{year}_dollar_prices.csv', sep=';', index=False)


if __name__ == '__main__':
  main()
