import pandas as pd
import numpy as np
import os

base_input_folder = "new_raw_data"
output_folder = "clean_data_2"
calls_output_folder = os.path.join(output_folder, "calls")
puts_output_folder = os.path.join(output_folder, "puts")
os.makedirs(calls_output_folder, exist_ok=True)
os.makedirs(puts_output_folder, exist_ok=True)

def clean_data(df):
    df.columns = df.columns.str.strip().str.replace(r'[\[\]\s]', '', regex=True)
    numeric_cols = [
        'UNDERLYING_LAST', 'DTE', 'C_DELTA', 'C_GAMMA', 'C_VEGA', 'C_THETA',
        'C_RHO', 'C_IV', 'C_VOLUME', 'C_LAST', 'C_BID', 'C_ASK', 'STRIKE',
        'P_BID', 'P_ASK', 'P_LAST', 'P_DELTA', 'P_GAMMA', 'P_VEGA', 'P_THETA',
        'P_RHO', 'P_IV', 'P_VOLUME'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'QUOTE_UNIXTIME' in df.columns:
        df['QUOTE_UNIXTIME'] = pd.to_datetime(df['QUOTE_UNIXTIME'], unit='s', errors='coerce')
    if 'EXPIRE_DATE' in df.columns:
        df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'], errors='coerce')

    df = df.dropna(subset=['QUOTE_UNIXTIME'])

    def split_size(size_str):
        if pd.isna(size_str) or not isinstance(size_str, str):
            return [np.nan, np.nan]
        try:
            return [int(s) for s in size_str.split(' x ')]
        except ValueError:
            return [np.nan, np.nan]

    if 'C_SIZE' in df.columns:
        df[['C_BID_SIZE', 'C_ASK_SIZE']] = df['C_SIZE'].apply(split_size).apply(pd.Series)
        df.drop(columns=['C_SIZE'], inplace=True, errors='ignore')

    if 'P_SIZE' in df.columns:
        df[['P_BID_SIZE', 'P_ASK_SIZE']] = df['P_SIZE'].apply(split_size).apply(pd.Series)
        df.drop(columns=['P_SIZE'], inplace=True, errors='ignore')

    df = df[(df['C_BID'] >= 0) & (df['C_ASK'] >= 0) & (df['C_ASK'] >= df['C_BID'])]
    df = df[(df['P_BID'] >= 0) & (df['P_ASK'] >= 0) & (df['P_ASK'] >= df['P_BID'])]

    df = df.dropna(subset=['UNDERLYING_LAST', 'STRIKE', 'DTE'])

    df['C_VOLUME'] = df.get('C_VOLUME', 0).fillna(0)
    df['P_VOLUME'] = df.get('P_VOLUME', 0).fillna(0)

    df['C_LAST'] = df['C_LAST'].fillna((df['C_BID'] + df['C_ASK']) / 2)
    df['P_LAST'] = df['P_LAST'].fillna((df['P_BID'] + df['P_ASK']) / 2)

    df = df.dropna(subset=['C_BID', 'C_ASK', 'P_BID', 'P_ASK'], how='all')

    df['MidPrice_Call'] = (df['C_BID'] + df['C_ASK']) / 2
    df['MidPrice_Put'] = (df['P_BID'] + df['P_ASK']) / 2
    df['Bid_Ask_Spread_Call'] = (df['C_ASK'] - df['C_BID']) / df['MidPrice_Call'].replace(0, np.nan)
    df['Bid_Ask_Spread_Put'] = (df['P_ASK'] - df['P_BID']) / df['MidPrice_Put'].replace(0, np.nan)

    df['TimeToExpiryYears'] = df['DTE'] / 365

    df = df[(df['C_IV'] > 0) & (df['C_IV'] < 10) | df['C_IV'].isna()]
    df = df[(df['P_IV'] > 0) & (df['P_IV'] < 10) | df['P_IV'].isna()]
    df = df[(df['C_LAST'] < df['UNDERLYING_LAST'] * 2)]
    df = df[(df['P_LAST'] < df['UNDERLYING_LAST'] * 2)]
    df = df[df['DTE'] >= 7]

    if 'C_BID_SIZE' in df.columns and 'C_ASK_SIZE' in df.columns:
        df = df[(df['C_BID_SIZE'] > 0) | (df['C_ASK_SIZE'] > 0)]
    if 'P_BID_SIZE' in df.columns and 'P_ASK_SIZE' in df.columns:
        df = df[(df['P_BID_SIZE'] > 0) | (df['P_ASK_SIZE'] > 0)]

    df = df.drop_duplicates()

    return df


def split_call_put_datasets(df):
    df_calls = df.dropna(subset=['C_IV']).copy()
    call_columns = [
        'QUOTE_UNIXTIME', 'EXPIRE_DATE', 'UNDERLYING_LAST', 'STRIKE', 'DTE',
        'C_BID', 'C_ASK', 'C_LAST', 'C_VOLUME',
        'C_DELTA', 'C_GAMMA', 'C_VEGA', 'C_THETA', 'C_RHO',
        'MidPrice_Call', 'Bid_Ask_Spread_Call',
        'TimeToExpiryYears', 'C_IV'
    ]
    df_calls = df_calls[[col for col in call_columns if col in df_calls.columns]]

    df_puts = df.dropna(subset=['P_IV']).copy()
    put_columns = [
        'QUOTE_UNIXTIME', 'EXPIRE_DATE', 'UNDERLYING_LAST', 'STRIKE', 'DTE',
        'P_BID', 'P_ASK', 'P_LAST', 'P_VOLUME',
        'P_DELTA', 'P_GAMMA', 'P_VEGA', 'P_THETA', 'P_RHO',
        'MidPrice_Put', 'Bid_Ask_Spread_Put',
        'TimeToExpiryYears', 'P_IV'
    ]
    df_puts = df_puts[[col for col in put_columns if col in df_puts.columns]]

    return df_calls, df_puts


for year in range(2010, 2024):
    year_folder = os.path.join(base_input_folder, f"SPX EOD {year}")
    if not os.path.exists(year_folder):
        print(f"Folder not found: {year_folder}")
        continue

    for month in range(1, 13):
        month_str = f"{month:02d}"
        file_name = f"spx_eod_{year}{month_str}.txt"
        file_path = os.path.join(year_folder, file_name)

        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path, low_memory=False)
        cleaned_df = clean_data(df)
        df_calls, df_puts = split_call_put_datasets(cleaned_df)

        call_output_path = os.path.join(calls_output_folder, f"calls_spx_eod_{year}{month_str}.pkl")
        put_output_path = os.path.join(puts_output_folder, f"puts_spx_eod_{year}{month_str}.pkl")

        df_calls.to_pickle(call_output_path)
        df_puts.to_pickle(put_output_path)

        print(f"Cleaned {file_name}")

print("Finished cleaning all data")