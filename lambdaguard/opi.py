import numpy as np

def overfitting_index(model, X, y, noise_std=1e-3, eps=1e-8):
    """
    Computes the Overfitting Index (OFI)
    Works for regression models (GradientBoosting) or classification.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            preds = proba[:, 1]
            y_cont = y
        else:
            preds = np.max(proba, axis=1)
            y_cont = (np.argmax(proba, axis=1) == y).astype(float)
    else:
        preds = model.predict(X)
        y_cont = y

    A = np.corrcoef(preds, y_cont)[0, 1] if np.std(preds) > eps and np.std(y_cont) > eps else 0
    C = np.var(preds)

    noise = np.random.normal(0, noise_std, X.shape)
    if hasattr(model, "predict_proba"):
        proba_noisy = model.predict_proba(X + noise)
        preds_noisy = proba_noisy[:, 1] if proba_noisy.shape[1]==2 else np.max(proba_noisy, axis=1)
    else:
        preds_noisy = model.predict(X + noise)

    S = np.mean(np.abs(preds - preds_noisy)) / (np.std(preds) + eps)
    OFI = (C / (A + C + eps)) * S

    return {"OFI": OFI, "Alignment": A, "Capacity": C, "Instability": S}
