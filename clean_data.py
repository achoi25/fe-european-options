import pandas as pd
import numpy as np
import os

for year in range(2023, 2024):
    for quarter in range(4):
        for i in range(3):
            prefix = "0" * (quarter != 3)
            path = f"raw_data/spx_eod_{year}q{quarter + 1}/"
            file_name = f"spx_eod_{year}{prefix}{3 * quarter + i + 1}.txt"
            df = pd.read_csv(path + file_name)

            df.columns = df.columns.str.replace(r'[\[\]\s]', '', regex=True)

            numeric_cols = [
                'UNDERLYING_LAST', 'DTE', 'C_DELTA', 'C_GAMMA', 'C_VEGA', 'C_THETA', 
                'C_RHO', 'C_IV', 'C_VOLUME', 'C_LAST', 'C_BID', 'C_ASK', 'STRIKE',
                'P_BID', 'P_ASK', 'P_LAST', 'P_DELTA', 'P_GAMMA', 'P_VEGA', 'P_THETA', 
                'P_RHO', 'P_IV', 'P_VOLUME', 'STRIKE_DISTANCE', 'STRIKE_DISTANCE_PCT'
            ]

            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['QUOTE_UNIXTIME'] = pd.to_datetime(df['QUOTE_UNIXTIME'], unit='s')
            df['QUOTE_READTIME'] = pd.to_datetime(df['QUOTE_READTIME'])
            df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'])

            def split_size(size_str):
                if pd.isna(size_str) or not isinstance(size_str, str):
                    return [np.nan, np.nan]
                try:
                    return [int(s) for s in size_str.split(' x ')]
                except ValueError:
                    return [np.nan, np.nan]

            df[['C_BID_SIZE', 'C_ASK_SIZE']] = df['C_SIZE'].apply(split_size).apply(pd.Series)
            df[['P_BID_SIZE', 'P_ASK_SIZE']] = df['P_SIZE'].apply(split_size).apply(pd.Series)

            df = df.drop(columns=['C_SIZE', 'P_SIZE'])

            output_folder = 'clean_data'
            output_file_name = 'cleaned_' + file_name.replace('.txt', '.csv')
            output_path = os.path.join(output_folder, output_file_name)

            os.makedirs(output_folder, exist_ok=True)

            df.to_csv(output_path, index=False)

            print(f"Data cleaned and saved to {output_path}")