<p align="center">
  <img src="docs/logo.png" alt="λ-Guard" width="160"/>
</p>

<p align="center">
<strong>Overfitting detection for Gradient Boosting</strong> — <em>no validation set required</em><br>
<i>Detect the moment when your model stops learning signal and starts memorizing structure.</i>
</p>

<p align="center">
  <a href="https://github.com/faberBI/lambdaguard/actions/workflows/tests.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/faberBI/lambdaguard/tests.yml?branch=main&logo=github" alt="Tests Status">
  </a>
  <a href="https://coveralls.io/github/faberBI/lambdaguard">
    <img src="https://img.shields.io/coveralls/github/faberBI/lambdaguard/main.svg" alt="Coverage Status">
  </a>
  <a href="https://pypi.org/project/lambda-guard-boosting/">
    <img src="https://img.shields.io/pypi/v/lambdaguard?logo=python" alt="PyPI Version">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License MIT">
  </a>
</p>

---

## ❓ Why λ-Guard?

In Gradient Boosting, overfitting often appears **before the validation error rises**.  
By that point, the model is already:

- ✂️ Splitting features into extremely fine regions  
- 🍃 Fitting leaves supported by very few observations  
- 🌪 Sensitive to tiny perturbations  

It’s **no longer improving predictions**, it’s **memorizing the training dataset**.  
**λ-Guard detects that moment automatically.**

---

## 🧠 Core Intuition

A boosting model learns two things simultaneously:

| Component | Role |
|-----------|------|
| Geometry  | partitions the feature space |
| Predictor | assigns values to each region |

Overfitting occurs when:

*"Geometry keeps growing, but predictor stops extracting real information."*

λ-Guard measures three key signals:

- 📦 **Capacity** → structural complexity  
- 🎯 **Alignment** → extracted signal  
- 🌊 **Stability** → fragility of predictions

---

## 🧩 Representation Matrix

Every tree divides the feature space into **leaves**.  
We record where each observation falls:
Z[i,j] = 1 if sample i falls in leaf j
Z[i,j] = 0 otherwise

- Rows → observations  
- Columns → leaves across all trees  

Think of **Z** as the **representation learned by the ensemble**.

- Linear regression → hat matrix **H**  
- Boosting → representation **Z**

---

## 📦 Capacity — Structural Complexity

- 🔹 Low C → few effective regions  
- 🔹 High C → model fragments space  

Late-stage boosting **increases C quickly**, often without improving predictions.

---

## 🎯 Alignment — Useful Information

- 🔹 High A → trees add real predictive signal  
- 🔹 Low A → trees mostly refine boundaries  

*"After some trees, alignment saturates."*  
Boosting continues **growing structure** even if prediction stops improving.

---

## 🌊 Stability — Sensitivity to Perturbations

- 🔹 Low S → smooth, robust model  
- 🔹 High S → brittle, sensitive model  

**Stability is the first signal to explode during overfitting.**

---

## 🔥 The Overfitting Index λ

| Situation | λ |
|-----------|---|
| Compact structure + stable predictions | low |
| Many regions + weak signal | high |
| Unstable predictions | very high |

**Interpretation:** measures how much structural complexity is wasted.  
Normalized λ ∈ [0,1] can be used to **compare models**.


## 🧪 Structural Overfitting Test

Detect if a few training points dominate the model using **approximate leverage**:
H_ii ≈ Σ_trees (learning_rate / leaf_size)
T1 = mean(H_ii) # global complexity
T2 = max(H_ii)/mean(H_ii) # local memorization


**Bootstrap procedure:**

1. Repeat B times: resample training data, recompute T1 & T2  
2. Compute p-values:  
   - p1 = P(T1_boot ≥ T1_obs)  
   - p2 = P(T2_boot ≥ T2_obs)  

Reject structural stability if:

p1 < α OR p2 < α


---

## 📊 What λ-Guard Distinguishes

| Regime | Meaning |
|--------|---------|
| ✅ Stable | smooth generalization |
| 📈 Global overfitting | too many effective parameters |
| ⚠️ Local memorization | few points dominate |
| 💥 Extreme | interpolation behavior |

---

## 🧭 When to Use

- Monitor boosting during training  
- Hyperparameter tuning  
- Small datasets (no validation split)  
- Diagnose late-stage performance collapse

---

## ⚙️ Installation

Install via GitHub:

```bash
pip install git+https://github.com/faberBI/lambdaguard.git

from sklearn.ensemble import GradientBoostingRegressor
from lambdaguard.ofi import generalization_index, instability_index, 
from lambdaguard.lambda_guard import lambda_guard_test, interpret
from lambdaguard.cusum import lambda_detect
import pandas as pd

# Fit a model
model = GradientBoostingRegressor(n_estimators=50, max_depth=3)
model.fit(X_train, y_train)

# Generalization index
GI, A, C = overfitting_index(model, X_train, y_train)
print('Generalization index: ", GI)

# Lambda-guard test
lg_res = lambda_guard_test(model, X_train)
print(interpret(lg_res))

# CUSUM-based detection
df = pd.DataFrame([
    {"model": "GBR", "n_estimators": 50, "max_depth": 3, "A": 0.8, "OFI_norm": 0.2},
    {"model": "GBR", "n_estimators": 100, "max_depth": 5, "A": 0.85, "OFI_norm": 0.3},
])
cusum_res = lambda_detect(
    df,
    model_name,
    complexity_metric="combined",
    lambda_col="OFI_norm",
    alignment_col="A",
    smooth_window=3,
    cusum_threshold_factor=1.5,
    baseline_points=10
)

```

## 📜 Citation

If you use **λ-Guard** in your research or projects, please cite the following:

**Fabrizio Di Sciorio, PhD**  
*Universidad de Almeria — Business and Economics Department*  
> "λ-Guard: Structural Overfitting Detection for Gradient Boosting Models"  

