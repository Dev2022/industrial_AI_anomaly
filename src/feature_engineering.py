import pandas as pd

from config.config import *

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def create_features(df):

    for col in SENSOR_COLS:

        df[f"{col}_mean_60"] = (

            df.groupby("machine_id")[col]
            .rolling(WINDOW_SIZE)
            .mean()
            .reset_index(0, drop=True)
        )

        df[f"{col}_std_60"] = (

            df.groupby("machine_id")[col]
            .rolling(WINDOW_SIZE)
            .std()
            .reset_index(0, drop=True)
        )

        df[f"{col}_max_60"] = (

            df.groupby("machine_id")[col]
            .rolling(WINDOW_SIZE)
            .max()
            .reset_index(0, drop=True)
        )

        df[f"{col}_delta"] = (

            df.groupby("machine_id")[col]
            .diff()
        )

    df.bfill(inplace=True)

    return df