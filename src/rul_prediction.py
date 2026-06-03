from xgboost import XGBRegressor

# ==========================================================
# RUL PREDICTION
# ==========================================================

def run_rul_prediction(df, X):

    df["RUL"] = (

        1000
        - (
            df["predicted_failure_probability"] * 900
        )
    ).clip(lower=1)

    rul_model = XGBRegressor(

        n_estimators=100,
        max_depth=5,
        learning_rate=0.05
    )

    rul_model.fit(X, df["RUL"])

    df["predicted_RUL"] = rul_model.predict(X)

    return df, rul_model