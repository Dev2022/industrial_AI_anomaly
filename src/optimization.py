import pandas as pd

from pulp import *

# ==========================================================
# MILP OPTIMIZATION
# ==========================================================

def optimize_maintenance(df):

    latest_df = (

        df.sort_values("timestamp")
        .groupby("machine_id")
        .tail(1)
    )

    machines = latest_df["machine_id"].tolist()

    criticality = {}
    maintenance_cost = {}
    failure_cost = {}

    for _, row in latest_df.iterrows():

        m = row["machine_id"]

        fp = row["predicted_failure_probability"]

        if fp < 0.30:

            criticality[m] = 1
            maintenance_cost[m] = 4000
            failure_cost[m] = 30000

        elif fp < 0.70:

            criticality[m] = 3
            maintenance_cost[m] = 12000
            failure_cost[m] = 150000

        else:

            criticality[m] = 5
            maintenance_cost[m] = 25000
            failure_cost[m] = 450000

    failure_prob = {

        row["machine_id"]:
        row["predicted_failure_probability"]

        for _, row in latest_df.iterrows()
    }

    prob = LpProblem(
        "Predictive_Maintenance",
        LpMinimize
    )

    x = LpVariable.dicts(
        "maintenance",
        machines,
        cat="Binary"
    )

    prob += lpSum([

        maintenance_cost[m] * x[m]

        +

        failure_cost[m]
        * failure_prob[m]
        * (1 - x[m])

        for m in machines
    ])

    prob += lpSum([

        x[m]
        for m in machines

    ]) <= 8

    for m in machines:

        if failure_prob[m] > 0.92:

            prob += x[m] == 1

    prob.solve(PULP_CBC_CMD(msg=False))

    results = []

    for m in machines:

        action = int(x[m].varValue)

        residual_risk = (
            failure_prob[m] * (1 - action)
        )

        expected_failure_cost = (
            residual_risk * failure_cost[m]
        )

        results.append({

            "machine_id": m,
            "criticality": criticality[m],
            "failure_probability":
                round(failure_prob[m], 3),

            "maintenance_action": action,

            "residual_risk":
                round(residual_risk, 3),

            "expected_failure_cost":
                round(expected_failure_cost, 2)
        })

    results_df = pd.DataFrame(results)

    return results_df, prob