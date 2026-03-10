import os
import sys

# Define 7 distinct market periods (balanced: 3 bull, 2 bear, 2 chop)
PERIODS = [
    {"name": "Meme Stock Squeeze (High Vol Bull)", "start": "2021-01-15", "end": "2021-03-15"},
    {"name": "Post-COVID Rally (High Vol Bull)", "start": "2020-11-01", "end": "2020-12-31"},
    {"name": "2022 Rate Hikes (Med Vol Bear)", "start": "2022-05-01", "end": "2022-06-30"},
    {"name": "AI Boom Start (Med Vol Bull)", "start": "2023-05-01", "end": "2023-06-30"},
    {"name": "Late 2023 Correction (Chop/Bear)", "start": "2023-09-01", "end": "2023-10-31"},
    {"name": "2022 Crypto Crash (High Vol Bear)", "start": "2022-01-15", "end": "2022-03-15"},
    {"name": "Q3 2024 Election Chop (Sideways)", "start": "2024-07-01", "end": "2024-08-31"},
]

def fetch_all():
    script_path = os.path.join(os.path.dirname(__file__), "data_loader.py")
    
    for i, p in enumerate(PERIODS):
        print(f"\n[{i+1}/{len(PERIODS)}] Fetching: {p['name']} | {p['start']} to {p['end']}")
        cmd = f'{sys.executable} "{script_path}" --start {p["start"]} --end {p["end"]}'
        os.system(cmd)
        
    print(f"\nAll {len(PERIODS)} multi-environment datasets successfully downloaded!")

if __name__ == "__main__":
    fetch_all()
