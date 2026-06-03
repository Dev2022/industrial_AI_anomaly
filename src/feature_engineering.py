from config.config import *

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def create_features(df):

    # ------------------------------------------------------
    # SHORT WINDOW FEATURES
    # FOR ANOMALY DETECTION
    # ------------------------------------------------------

    for col in SENSOR_COLS:

        df[f"{col}_mean_10"] = (

            df.groupby("machine_id")[col]
            .rolling(ANOMALY_WINDOW)
            .mean()
            .reset_index(0, drop=True)
        )

        df[f"{col}_std_10"] = (

            df.groupby("machine_id")[col]
            .rolling(ANOMALY_WINDOW)
            .std()
            .reset_index(0, drop=True)
        )

        df[f"{col}_max_10"] = (

            df.groupby("machine_id")[col]
            .rolling(ANOMALY_WINDOW)
            .max()
            .reset_index(0, drop=True)
        )

        # delta

        df[f"{col}_delta"] = (

            df.groupby("machine_id")[col]
            .diff()
        )

    # ------------------------------------------------------
    # LONG WINDOW FEATURES
    # FOR FAILURE PREDICTION
    # ------------------------------------------------------

    for col in SENSOR_COLS:

        df[f"{col}_mean_60"] = (

            df.groupby("machine_id")[col]
            .rolling(FAILURE_WINDOW)
            .mean()
            .reset_index(0, drop=True)
        )

        df[f"{col}_std_60"] = (

            df.groupby("machine_id")[col]
            .rolling(FAILURE_WINDOW)
            .std()
            .reset_index(0, drop=True)
        )

        df[f"{col}_max_60"] = (

            df.groupby("machine_id")[col]
            .rolling(FAILURE_WINDOW)
            .max()
            .reset_index(0, drop=True)
        )

    df.bfill(inplace=True)

    return df