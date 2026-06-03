from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config.config import *

# ==========================================================
# ANOMALY DETECTION
# ==========================================================

def run_anomaly_detection(df):

    anomaly_features = []

    for col in SENSOR_COLS:

        anomaly_features.extend([

            f"{col}_mean_60",
            f"{col}_std_60",
            f"{col}_max_60",
            f"{col}_delta"
        ])

    scaler = StandardScaler()

    X_anomaly = scaler.fit_transform(
        df[anomaly_features]
    )

    iso = IsolationForest(

        n_estimators=300,
        contamination=0.05,
        random_state=42
    )

    df["anomaly_label"] = iso.fit_predict(
        X_anomaly
    )

    df["anomaly_label"] = df["anomaly_label"].map({

        1: 0,
        -1: 1
    })

    return df, anomaly_features