import panda as pd
import numpy as np
import glob
from datetime import datetime

def Load_and_Prepare_Data():
    files = glob.glob("D:\School\Clubs\FEC\FECProjects\fe-european-options-main\clean_data\*.csv")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    return df

def clean_option_data(df):
    # Remove duplicates and rows with missing critical values
    df = df.drop_duplicate()
    df = df.dropna(subset=["UNDERLYING_LAST", "STRIKE", "C_LAST", "P_LAST"])

    # Remove negative values
    df = df[(df["UNDERLYING_LAST"] > 0) & (df["STRIKE"] > 0) & (df["C_LAST"] >= 0) & (df["P_LAST"] >= 0)]
    df = df[(df["C_BID"] >= 0) & (df["C_ASK"] >= 0) & (df["P_BID"] >= 0) & (df["P_ASK"] >= 0)]
    
    # Days to expiration
    if "DTE" not in df.columns:
        df["DTE"] = (pd.to_datetime(df["EXPIRATION"]) - pd.to_datetime(df["QUOTE_DATE"])).dt.days

    return df

def add_features(df):

    # Find Moneyness and Log Moneyness
    df["moneyness"] = df["UNDERLYING_LAST"] / df["STRIKE"]
    df["log_moneyness"] = np.log(df["moneyness"])

    # Mid prices
    df["C_MID"] = (df["C_BID"] + df["C_ASK"]) / 2
    df["P_MID"] = (df["P_BID"] + df["P_ASK"]) / 2

    # Option spread (Liquidity)
    df["spread_C"] = df["C_ASK"] - df["C_BID"]
    df["rel_spread_C"] = df["spread_C"] / df["C_MID"]

    # Filter out extreme spreads (can be changed to see effect on model)
    df = df[df["C_IV"].between(0.01, 2.0)]

    return df

def select_features(df):
    useful_features = [
        "UNDERLYING_LAST",
        "STRIKE",
        "DTE",
        "moneyness",
        "C_IV",
        "P_DELTA",
        "C_GAMMA",
        "P_VEGA",
        "C_THETA",
        "C_RHO",
        "C_VOLUME",
        "C_BID",
        "C_ASK",
    ]

    target = "C_MID"

    df = df.dropna(subset=[target])
    X = df[useful_features]
    y = df[target]

    return X, y

def preprocess_data():
    df = Load_and_Prepare_Data()
    df = clean_option_data(df)
    df = add_features(df)
    X, y = select_features(df)
    return X, y, df