import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def boosting_leverage(model, X):
    n = X.shape[0]
    influence = np.zeros(n)
    for est in model.estimators_.ravel():
        leaf_id = est.apply(X)
        unique, counts = np.unique(leaf_id, return_counts=True)
        leaf_sizes = dict(zip(unique, counts))
        lr = model.learning_rate
        for i in range(n):
            influence[i] += lr / leaf_sizes[leaf_id[i]]
    return influence

def lambda_guard_test(model, X, B=300, alpha=0.05, plot=True):
    n = X.shape[0]
    H = boosting_leverage(model, X)
    T1_obs = H.sum() / n
    T2_obs = H.max() / H.mean()
    T1_boot = np.zeros(B)
    T2_boot = np.zeros(B)
    for b in range(B):
        idx = np.random.choice(n, n, replace=True)
        Hb = boosting_leverage(model, X[idx])
        T1_boot[b] = Hb.sum() / n
        T2_boot[b] = Hb.max() / Hb.mean()
    q1, q2 = np.quantile(T1_boot, 1-alpha), np.quantile(T2_boot, 1-alpha)
    p1, p2 = np.mean(T1_boot >= T1_obs), np.mean(T2_boot >= T2_obs)
    reject = (p1 < alpha) or (p2 < alpha)
    if plot:
        fig, axes = plt.subplots(1,2,figsize=(12,4))
        axes[0].hist(T1_boot, bins=30, density=True, alpha=0.7)
        axes[0].axvline(T1_obs, color="black", label="Observed")
        axes[0].axvline(q1, color="red", linestyle="--", label="Critical")
        axes[0].set_title("T1: Effective DoF ratio")
        axes[0].legend()
        axes[1].hist(T2_boot, bins=30, density=True, alpha=0.7)
        axes[1].axvline(T2_obs, color="black", label="Observed")
        axes[1].axvline(q2, color="red", linestyle="--", label="Critical")
        axes[1].set_title("T2: Peak leverage ratio")
        axes[1].legend()
        plt.show()
    return {
        "T1_df_ratio": T1_obs, "critical_df_ratio": q1, "p_df_ratio": p1,
        "T2_peak_ratio": T2_obs, "critical_peak_ratio": q2, "p_peak_ratio": p2,
        "reject_H0": reject
    }

def interpret(res):
    if not res["reject_H0"]:
        return "✔ REGIME STABILE / GENERALIZZANTE"
    if res["p_df_ratio"] < 0.05 and res["p_peak_ratio"] < 0.05:
        return "✖ REGIME INTERPOLANTE (OVERFITTING FORTE)"
    if res["p_df_ratio"] < 0.05:
        return "✖ COMPLESSITÀ GLOBALE ECCESSIVA"
    return "✖ (LEVERAGE SPIKES)"
