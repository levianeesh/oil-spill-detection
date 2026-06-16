import pandas as pd
import os

# --- PATHS ---
# We use your existing big file as input
INPUT_FILE = 'data/ais_data/vessel_data.csv' 
# We save a new, small, fast file
OUTPUT_FILE = 'data/ais_data/vessel_data_clean.csv'

# --- CONFIGURATION ---
# This must match the approximate area of your satellite image
# Example: Gulf of Mexico coordinates
TARGET_LAT = 28.5  
TARGET_LON = -90.5
SEARCH_RADIUS = 2.0 # Degrees (~200km window)

print(f"Reading large file: {INPUT_FILE}...")
print("This might take 1-2 minutes depending on file size...")

try:
    # 1. Load only necessary columns to save RAM
    cols = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'VesselName']
    
    # Check if file exists first
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, usecols=cols)

    print(f"Original Row Count: {len(df)}")

    # 2. Filter Data (Keep only ships near the spill)
    df_filtered = df[
        (df['LAT'] > TARGET_LAT - SEARCH_RADIUS) & 
        (df['LAT'] < TARGET_LAT + SEARCH_RADIUS) & 
        (df['LON'] > TARGET_LON - SEARCH_RADIUS) & 
        (df['LON'] < TARGET_LON + SEARCH_RADIUS)
    ]

    # 3. Save the optimized file
    df_filtered.to_csv(OUTPUT_FILE, index=False)
    
    print("------------------------------------------------")
    print(f"SUCCESS! Filtered data saved to: {OUTPUT_FILE}")
    print(f"Reduced from {len(df)} rows -> to {len(df_filtered)} rows.")
    print("Use 'vessel_data_clean.csv' in your final project.")
    print("------------------------------------------------")

except Exception as e:
    print(f"ERROR: {e}")