import numpy as np
import pandas as pd
from lambdaguard.cusum import lambda_detect

def test_lambda_detect_basic():
    # Fake dataset with 2 models and increasing OFI
    df = pd.DataFrame({
        "model": ["GBR"]*5,
        "dataset": ["test"]*5,
        "min_samples_leaf": [2]*5,
        "n_estimators": [10,20,30,40,50],
        "max_depth": [3]*5,
        "A": np.linspace(0.8, 0.95, 5),
        "C": np.linspace(0.1,0.5,5),
        "OFI": np.linspace(0.1,0.6,5),
        "OFI_norm": np.linspace(0.1,0.6,5)
    })

    result = lambda_detect(df, "GBR")
    assert "overfitting_detected" in result
    assert "best_model_before_overfitting" in result
