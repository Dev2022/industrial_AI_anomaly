import random
import numpy as np
import pandas as pd

from config.config import *

# ==========================================================
# DATA GENERATION
# ==========================================================

def generate_dataset():

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    rows = []

    for machine_id in range(NUM_MACHINES):

        degradation = 0

        if machine_id in HEALTHY_MACHINES:

            degradation_rate = 0.00005
            anomaly_probability = 0.002

            base_temp = 68
            base_vibration = 0.18
            base_acoustic = 28

        elif machine_id in MODERATE_MACHINES:

            degradation_rate = 0.0007
            anomaly_probability = 0.015

            base_temp = 75
            base_vibration = 0.28
            base_acoustic = 42

        else:

            degradation_rate = 0.0018
            anomaly_probability = 0.06

            base_temp = 85
            base_vibration = 0.50
            base_acoustic = 65

        anomaly_active = False
        anomaly_duration = 0

        for t in range(TIME_STEPS):

            degradation += np.random.uniform(
                degradation_rate * 0.5,
                degradation_rate * 1.5
            )

            degradation = min(degradation, 1)

            temperature = (
                base_temp
                + degradation * 60
                + np.sin(t / 40) * 2
                + np.random.normal(0, 2)
            )

            vibration = (
                base_vibration
                + degradation * 1.0
                + np.sin(t / 25) * 0.03
                + np.random.normal(0, 0.03)
            )

            acoustic = (
                base_acoustic
                + degradation * 100
                + np.random.normal(0, 4)
            )

            pressure = (
                120
                - degradation * 35
                + np.random.normal(0, 2)
            )

            rpm = (
                3000
                - degradation * 350
                + np.random.normal(0, 45)
            )

            current = (
                40
                + degradation * 28
                + np.random.normal(0, 2)
            )

            anomaly_spike = 0

            if (
                not anomaly_active
                and np.random.rand() < anomaly_probability
            ):

                anomaly_active = True

                anomaly_duration = np.random.randint(15, 60)

                anomaly_type = random.choice([

                    "bearing_failure",
                    "overheating",
                    "misalignment",
                    "pressure_leak",
                    "motor_instability"
                ])

            if anomaly_active:

                anomaly_spike = 1

                anomaly_duration -= 1

                if anomaly_type == "bearing_failure":

                    vibration += np.random.uniform(0.4, 1.2)
                    acoustic += np.random.uniform(30, 70)
                    current += np.random.uniform(5, 12)

                elif anomaly_type == "overheating":

                    temperature += np.random.uniform(20, 45)
                    current += np.random.uniform(5, 15)

                elif anomaly_type == "misalignment":

                    vibration += np.random.uniform(0.3, 0.8)
                    rpm -= np.random.uniform(200, 500)

                elif anomaly_type == "pressure_leak":

                    pressure -= np.random.uniform(15, 35)
                    current += np.random.uniform(3, 8)

                elif anomaly_type == "motor_instability":

                    rpm -= np.random.uniform(300, 800)
                    vibration += np.random.uniform(0.2, 0.6)
                    acoustic += np.random.uniform(15, 40)

                if anomaly_duration <= 0:
                    anomaly_active = False

            rows.append({

                "timestamp": t,
                "machine_id": machine_id,

                "temperature": round(temperature, 2),
                "vibration": round(vibration, 4),
                "acoustic_emission": round(acoustic, 2),
                "pressure": round(pressure, 2),
                "rpm": round(rpm, 2),
                "current": round(current, 2),

                "hidden_degradation": round(degradation, 4),
                "hidden_anomaly": anomaly_spike
            })

    df = pd.DataFrame(rows)

    return df