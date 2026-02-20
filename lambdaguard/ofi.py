# ============================================================
# FULL EXPERIMENT: LAMBDA GUARD - MULTI-MODEL (OPTIMIZED)
# ============================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# -----------------------------
# GENERALIZATION COMPONENTS
# -----------------------------
def generalization_index(model, X, y):
    preds = model.predict(X)
    A = np.corrcoef(preds, y)[0, 1] if np.std(preds) > 0 else 0
    C = np.var(preds)
    GI = A / C if C > 0 else 0
    return GI, A, C

def instability_index(model, X, noise_std=1e-3, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, X.shape)
    preds_clean = model.predict(X)
    preds_noisy = model.predict(X + noise)
    S = np.mean(np.abs(preds_clean - preds_noisy)) / (np.std(preds_clean) + 1e-8)
    return S

# -----------------------------
# MODEL FACTORY
# -----------------------------
def create_model(model_name, n_estimators=100, max_depth=3, learning_rate=0.05, min_samples_leaf=1):
    if model_name == "GBR":
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    elif model_name == "XGB":
        return xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=42,
            verbosity=0
        )
    elif model_name == "LGBM":
        return lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            min_child_samples=min_samples_leaf,
            random_state=42,
            verbose=-1
        )
    elif model_name == "CAT":
        return CatBoostRegressor(
            iterations=n_estimators,
            depth=max_depth,
            learning_rate=learning_rate,
            random_seed=42,
            verbose=0
        )
    else:
        raise ValueError(f"Unknown model {model_name}")

# -----------------------------
# EXPERIMENT FUNCTION
# -----------------------------
def run_experiment_multi_model(X, y, dataset_name, model_names=["GBR"], 
                               n_estimators_list=[50,100,200], max_depth_list=[3,5,7], min_samples_leaf_list=[2,5]):
    print(f"\n{'='*70}\nDATASET: {dataset_name}\n{'='*70}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    results = []

    for model_name in model_names:
        for sl, n_est, depth in product(min_samples_leaf_list, n_estimators_list, max_depth_list):
            model = create_model(model_name, n_estimators=n_est, max_depth=depth, min_samples_leaf=sl)
            model.fit(X_train, y_train)

            GI, A, C = generalization_index(model, X_train, y_train)
            S = instability_index(model, X_train)
            OFI = (C / (A + C)) * S
            G_norm = A / (A + C)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            gap = rmse_test - rmse_train

            results.append({
                "model": model_name,
                "dataset": dataset_name,
                "min_samples_leaf": sl,
                "n_estimators": n_est,
                "max_depth": depth,
                "A": A,
                "C": C,
                "GI": GI,
                "G_norm": G_norm,
                "Instability": S,
                "OFI": OFI,
                "Train_RMSE": rmse_train,
                "Test_RMSE": rmse_test,
                "Gap": gap
            })

    df = pd.DataFrame(results)
    df["OFI_norm"] = df.groupby("model")["OFI"].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    return df

# -----------------------------
# PLOT FUNCTION
# -----------------------------
def plot_all_multi_model(df, metric="Gap"):
    for model_name in df["model"].unique():
        df_model = df[df["model"] == model_name]
        for x_col in ["G_norm", "OFI_norm"]:
            plt.figure(figsize=(6,5))
            sns.regplot(data=df_model, x=x_col, y=metric)
            plt.title(f"{model_name} - {x_col} vs {metric}")
            plt.grid(True)
            plt.show()


# -----------------------------
# REGRESSION TEST
# -----------------------------
def regression_test(df):
  X = df['OFI_norm']
  y = df['Gap']
  X_const = sm.add_constant(X)
  model = sm.OLS(y, X_const).fit()
  print(model.summary())
  plt.figure(figsize=(8,6))
  plt.scatter(df['OFI_norm'], df_model['Gap'], alpha=0.6)
  plt.plot(df['OFI_norm'], model.predict(X_const), color='red', linewidth=2)
  plt.xlabel('Lambda')
  plt.ylabel('RMSE gap (test - train)')
  plt.title(f'Regression Gap RMSE vs Lambda - {df_model["model"].iloc[0]}')
  plt.grid(True)
  plt.show()
