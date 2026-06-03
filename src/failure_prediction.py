import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier

# ==========================================================
# FAILURE PREDICTION
# ==========================================================

def run_failure_prediction(df, feature_cols):

    failure_score = (

        0.25 * (df["temperature_mean_60"] / 120)

        +

        0.25 * (df["vibration_mean_60"] / 1.5)

        +

        0.20 * (df["acoustic_emission_mean_60"] / 150)

        +

        0.10 * (1 - (df["pressure_mean_60"] / 120))

        +

        0.10 * (1 - (df["rpm_mean_60"] / 3000))

        +

        0.10 * df["anomaly_label"]
    )

    df["failure_probability"] = failure_score.clip(0, 1)

    df["failure_within_7_days"] = (

        df["failure_probability"]
        > np.random.uniform(0.45, 0.75, len(df))

    ).astype(int)

    X = df[feature_cols]
    y = df["failure_within_7_days"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(

        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=2,
        min_child_weight=4,
        reg_alpha=0.5,
        reg_lambda=2,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {

        "accuracy":
            accuracy_score(y_test, preds),

        "precision":
            precision_score(y_test, preds),

        "recall":
            recall_score(y_test, preds),

        "f1":
            f1_score(y_test, preds)
    }

    print(classification_report(y_test, preds))

    df["predicted_failure_probability"] = (
        model.predict_proba(X)[:, 1]
    )

    return df, model, X_test, metrics