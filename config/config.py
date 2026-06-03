# ==========================================================
# CONFIGURATION
# ==========================================================

NUM_MACHINES = 20
TIME_STEPS = 1500
ANOMALY_WINDOW = 10
FAILURE_WINDOW = 60

RANDOM_SEED = 42

RESULTS_DIR = "results"

SENSOR_COLS = [

    "temperature",
    "vibration",
    "acoustic_emission",
    "pressure",
    "rpm",
    "current"
]

HEALTHY_MACHINES = [0,1,2,3,4,5,6,7]

MODERATE_MACHINES = [8,9,10,11,12,13,14,15]

CRITICAL_MACHINES = [16,17,18,19]