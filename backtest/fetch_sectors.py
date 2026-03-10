import pandas as pd
import json
import os

def generate_sp500_sectors():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        tables = pd.read_html(url, storage_options={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        df = tables[0]

        
        # Create a mapping dictionary {Ticker: Sector}
        sector_map = {}
        for index, row in df.iterrows():
            ticker = row['Symbol'].replace('.','-') # Fix BRK.B -> BRK-B format if needed
            sector = row['GICS Sector']
            sector_map[ticker] = sector
            
        # Also add a few ETFs that the strategy trades
        sector_map['SPY'] = 'Market ETF'
        sector_map['QQQ'] = 'Market ETF'
        sector_map['TQQQ'] = 'Leveraged ETF'
        sector_map['SQQQ'] = 'Leveraged ETF'
        sector_map['PSQ'] = 'Inverse ETF'
        sector_map['QID'] = 'Inverse ETF'
        sector_map['SDS'] = 'Inverse ETF'
        sector_map['SPXU'] = 'Inverse ETF'
        sector_map['SH'] = 'Inverse ETF'
        sector_map['DOG'] = 'Inverse ETF'
        sector_map['DXD'] = 'Inverse ETF'
        sector_map['SDOW'] = 'Inverse ETF'
            
        output_path = os.path.join(os.path.dirname(__file__), 'sp500_sectors.json')
        with open(output_path, 'w') as f:
            json.dump(sector_map, f, indent=4)
            
        print(f"Successfully generated sp500_sectors.json with {len(sector_map)} entries.")
    except Exception as e:
        print(f"Failed to fetch or parse S&P 500 data. Error type: {type(e)}")

if __name__ == "__main__":
    generate_sp500_sectors()
