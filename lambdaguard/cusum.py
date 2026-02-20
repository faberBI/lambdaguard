import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def detect_structural_overfitting_cusum_robust(
    df,
    model_name,
    complexity_metric="combined",
    lambda_col="OFI_norm",
    alignment_col="A",
    smooth_window=3,
    cusum_threshold_factor=1.5,
):
    df_model = df[df["model"] == model_name].copy()
    if complexity_metric == "combined":
        df_model["complexity"] = df_model["n_estimators"] * df_model["max_depth"]
    else:
        df_model["complexity"] = df_model[complexity_metric]
    df_model = df_model.sort_values("complexity")
    lambdas = df_model[lambda_col].values
    alignment = df_model[alignment_col].values
    complexity = df_model["complexity"].values

    delta_lambda = pd.Series(np.diff(lambdas)).rolling(smooth_window, min_periods=1).mean().values
    delta2_lambda = pd.Series(np.diff(delta_lambda)).rolling(smooth_window, min_periods=1).mean().values

    mean_d2 = np.mean(delta2_lambda)
    std_d2 = np.std(delta2_lambda)
    centered_d2 = delta2_lambda - mean_d2
    cusum = np.zeros_like(centered_d2)
    for i in range(1, len(centered_d2)):
        cusum[i] = max(0, cusum[i-1] + centered_d2[i])
    cusum_threshold = cusum_threshold_factor * std_d2

    change_index = None
    delta_alignment = np.diff(alignment)
    for i, val in enumerate(cusum):
        align_flat = delta_alignment[i] < 0.01 if i < len(delta_alignment) else False
        if val > cusum_threshold and align_flat:
            change_index = i + 2
            break

    plt.figure(figsize=(10,6))
    plt.plot(complexity, lambdas, '-o', label='λ (OFI_norm)')
    plt.plot(complexity[1:], delta_lambda, '-s', label='Δλ')
    plt.plot(complexity[2:], delta2_lambda, '-^', label='Δ²λ')
    plt.plot(complexity[2:], cusum, '-x', label='CUSUM Δ²λ', alpha=0.7)
    if change_index is not None:
        plt.axvline(complexity[change_index], color='red', linestyle='--', label='Change Point')
        plt.scatter(complexity[change_index], lambdas[change_index], color='red', s=100)
    plt.xlabel("Complexity (n_estimators*max_depth)")
    plt.ylabel("λ / Δλ / Δ²λ / CUSUM")
    plt.title(f"Structural Overfitting Detection (CUSUM) - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    if change_index is None:
        return {"overfitting_detected": False, "message": "No structural acceleration detected"}

    return {
        "overfitting_detected": True,
        "complexity_at_change": complexity[change_index],
        "lambda_at_change": lambdas[change_index],
        "delta_lambda_at_change": delta_lambda[change_index-1],
        "delta2_lambda_at_change": delta2_lambda[change_index-2],
        "cusum_at_change": cusum[change_index-2],
        "cusum_threshold": cusum_threshold,
        "complexity": complexity,
        "lambdas": lambdas,
        "delta_lambda": delta_lambda,
        "delta2_lambda": delta2_lambda,
        "cusum": cusum,
        "change_index": change_index
    }
