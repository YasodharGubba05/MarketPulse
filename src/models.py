"""Regression, classification, LSTM, GARCH, volatility ML.

Models:
- Linear Regression
- Random Forest Regressor / Classifier
- XGBoost Regressor / Classifier  (falls back to sklearn GBM when libomp missing)
- LightGBM Regressor / Classifier (falls back gracefully when not installed)
- Stacking Ensemble Regressor
- GARCH(p,q) volatility via arch
- LSTM (Keras/TF first, then PyTorch fallback)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingRegressor,
    StackingClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy XGBoost
# ---------------------------------------------------------------------------
_xgb_initialized = False
_XGB_OK = False
_XGBClassifier: Any = None
_XGBRegressor: Any = None


def _ensure_xgb() -> bool:
    global _xgb_initialized, _XGB_OK, _XGBClassifier, _XGBRegressor
    if _xgb_initialized:
        return _XGB_OK
    _xgb_initialized = True
    try:
        from xgboost import XGBClassifier, XGBRegressor as XGBR
        _XGBClassifier = XGBClassifier
        _XGBRegressor = XGBR
        _XGB_OK = True
    except Exception as e:
        err = str(e)
        if "libomp" in err or "OpenMP" in err:
            logger.warning(
                "XGBoost skipped: OpenMP (libomp) not found — using sklearn GradientBoosting instead. "
                "To enable XGBoost on macOS: brew install libomp"
            )
        else:
            logger.warning("XGBoost unavailable (%s); using sklearn GradientBoosting.", err[:300])
        _XGB_OK = False
    return _XGB_OK


# ---------------------------------------------------------------------------
# Lazy LightGBM
# ---------------------------------------------------------------------------
_lgbm_initialized = False
_LGBM_OK = False
_LGBMRegressor: Any = None
_LGBMClassifier: Any = None


def _ensure_lgbm() -> bool:
    global _lgbm_initialized, _LGBM_OK, _LGBMRegressor, _LGBMClassifier
    if _lgbm_initialized:
        return _LGBM_OK
    _lgbm_initialized = True
    try:
        import lightgbm as lgb
        _LGBMRegressor = lgb.LGBMRegressor
        _LGBMClassifier = lgb.LGBMClassifier
        _LGBM_OK = True
    except Exception as e:
        logger.warning("LightGBM unavailable (%s); ensemble will skip it.", str(e)[:200])
        _LGBM_OK = False
    return _LGBM_OK


# ---------------------------------------------------------------------------
# Lazy Keras / TF
# ---------------------------------------------------------------------------
_KERAS = None


def _keras():
    global _KERAS
    if _KERAS is False:
        return None
    if _KERAS is not None:
        return _KERAS
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        _KERAS = (keras, layers)
    except ImportError as e:
        logger.debug("TensorFlow/Keras not available (%s); will use PyTorch LSTM if torch is installed.", e)
        _KERAS = False
    return _KERAS if _KERAS is not False else None


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Regression models
# ---------------------------------------------------------------------------

def train_linear_regression(X_train, y_train):
    m = LinearRegression()
    m.fit(X_train, y_train)
    return m


def train_rf_regressor(X_train, y_train, random_state: int = 42):
    m = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    m.fit(X_train, y_train)
    return m


def train_xgb_regressor(X_train, y_train, random_state: int = 42, extra_params: dict | None = None):
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if extra_params:
        params.update(extra_params)
    if _ensure_xgb():
        m = _XGBRegressor(**params)
    else:
        m = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            random_state=random_state,
        )
    m.fit(X_train, y_train)
    return m


def train_lightgbm_regressor(X_train, y_train, random_state: int = 42,
                               extra_params: dict | None = None):
    """LightGBM regressor. Falls back to RF when LightGBM is not installed."""
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    if extra_params:
        params.update(extra_params)
    if _ensure_lgbm():
        m = _LGBMRegressor(**params)
    else:
        logger.warning("LightGBM not available — using RF regressor as fallback.")
        m = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1)
    m.fit(X_train, y_train)
    return m


def train_ensemble_regressor(X_train, y_train, random_state: int = 42):
    """
    Stacking ensemble: base = [Ridge, RF, XGB/GBM], meta = Ridge.
    Gracefully degrades if XGB/LGB unavailable.
    """
    base = [
        ("ridge", Ridge(alpha=1.0)),
        ("rf", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1)),
    ]
    if _ensure_xgb():
        base.append(("xgb", _XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=random_state, n_jobs=-1
        )))
    else:
        base.append(("gbm", GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.05, random_state=random_state
        )))
    if _ensure_lgbm():
        base.append(("lgbm", _LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=random_state,
            n_jobs=-1, verbose=-1
        )))
    meta = Ridge(alpha=1.0)
    m = StackingRegressor(estimators=base, final_estimator=meta, cv=3, n_jobs=-1)
    m.fit(X_train, y_train)
    return m


# ---------------------------------------------------------------------------
# Classification models
# ---------------------------------------------------------------------------

def train_logistic(X_train, y_train, random_state: int = 42):
    m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    m.fit(X_train, y_train)
    return m


def train_rf_classifier(X_train, y_train, random_state: int = 42):
    m = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    m.fit(X_train, y_train)
    return m


def train_xgb_classifier(X_train, y_train, random_state: int = 42, extra_params: dict | None = None):
    params = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": random_state,
        "n_jobs": -1,
        "eval_metric": "logloss",
    }
    if extra_params:
        params.update(extra_params)
    if _ensure_xgb():
        m = _XGBClassifier(**params)
    else:
        m = GradientBoostingClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            random_state=random_state,
        )
    m.fit(X_train, y_train)
    return m


def train_lightgbm_classifier(X_train, y_train, random_state: int = 42,
                                extra_params: dict | None = None):
    """LightGBM classifier for crash detection. Falls back to RF."""
    params = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    if extra_params:
        params.update(extra_params)
    if _ensure_lgbm():
        m = _LGBMClassifier(**params)
    else:
        logger.warning("LightGBM not available — using RF classifier as fallback.")
        m = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced",
                                   random_state=random_state, n_jobs=-1)
    m.fit(X_train, y_train)
    return m


def train_ensemble_classifier(X_train, y_train, random_state: int = 42):
    """Stacking ensemble classifier for crash detection."""
    base = [
        ("logistic", LogisticRegression(max_iter=1000, class_weight="balanced",
                                         random_state=random_state)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced",
                                       random_state=random_state, n_jobs=-1)),
    ]
    if _ensure_xgb():
        base.append(("xgb", _XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=random_state, n_jobs=-1
        )))
    if _ensure_lgbm():
        base.append(("lgbm", _LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, class_weight="balanced",
            random_state=random_state, n_jobs=-1, verbose=-1
        )))
    meta = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    m = StackingClassifier(estimators=base, final_estimator=meta, cv=3, n_jobs=-1)
    m.fit(X_train, y_train)
    return m


# ---------------------------------------------------------------------------
# Volatility models
# ---------------------------------------------------------------------------

def train_volatility_xgb(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """Predict absolute next-period volatility from features."""
    if _ensure_xgb():
        m = _XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=random_state, n_jobs=-1,
        )
    else:
        m = GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.05, random_state=random_state,
        )
    m.fit(X_train, y_train)
    return m


def fit_garch_volatility(returns: pd.Series, p: int = 1, q: int = 1) -> Any:
    """GARCH(p,q) on percentage returns. Lazy-imports `arch`."""
    try:
        from arch import arch_model
    except ImportError:
        logger.debug("Optional package 'arch' not installed; GARCH metrics skipped.")
        return None
    r = 100.0 * returns.dropna()
    if len(r) < 100:
        return None
    try:
        am = arch_model(r, vol="Garch", p=p, q=q, rescale=False)
        return am.fit(disp="off")
    except Exception as e:
        logger.debug("GARCH fit failed: %s", e)
        return None


def rolling_volatility_baseline(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)


# ---------------------------------------------------------------------------
# LSTM — Keras path
# ---------------------------------------------------------------------------

def build_lstm(input_shape: tuple[int, int], units: int = 64):
    k = _keras()
    if k is None:
        raise RuntimeError("Keras not installed")
    keras, layers = k
    inp = keras.Input(shape=input_shape)
    x = layers.LSTM(units, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 40,
    batch_size: int = 32,
    validation_split: float = 0.1,
):
    if len(X_train) < 10:
        raise ValueError("Not enough sequences for LSTM")
    if _keras() is None:
        raise RuntimeError("Keras not installed")
    model = build_lstm((X_train.shape[1], X_train.shape[2]))
    model.fit(
        X_train, y_train,
        epochs=epochs, batch_size=batch_size,
        validation_split=validation_split, verbose=0,
    )
    return model


# ---------------------------------------------------------------------------
# LSTM — PyTorch path
# ---------------------------------------------------------------------------

def build_lstm_torch_model(n_features: int, hidden: int = 64):
    """Stacked LSTM regressor — mirrors Keras architecture. Requires PyTorch."""
    import torch.nn as nn
    import torch.nn.functional as F

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
            self.dropout1 = nn.Dropout(0.2)
            self.lstm2 = nn.LSTM(hidden, 32, batch_first=True)
            self.dropout2 = nn.Dropout(0.1)
            self.fc1 = nn.Linear(32, 32)
            self.fc2 = nn.Linear(32, 1)

        def forward(self, x):
            o1, _ = self.lstm1(x)
            o1 = self.dropout1(o1)
            o2, _ = self.lstm2(o1)
            last = o2[:, -1, :]
            z = self.dropout2(last)
            z = F.relu(self.fc1(z))
            return self.fc2(z).squeeze(-1)

    return LSTMNet()


def preferred_lstm_backend() -> str:
    if _keras() is not None:
        return "keras"
    if _torch() is not None:
        return "torch"
    return "none"


def train_lstm_torch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 40,
    batch_size: int = 32,
    validation_split: float = 0.1,
    random_state: int = 42,
):
    """PyTorch stacked LSTM for Python 3.13+ (no TF wheel)."""
    if _torch() is None:
        raise RuntimeError("PyTorch is not installed; cannot train LSTM.")
    if len(X_train) < 10:
        raise ValueError("Not enough sequences for LSTM")
    import torch
    import torch.nn as nn

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    n, _, nfeat = X_train.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_lstm_torch_model(nfeat).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()

    n_val = max(1, int(n * validation_split)) if n > 20 else 0
    n_tr = n - n_val if n_val else n
    if n_tr < 5:
        n_tr, n_val = n, 0

    X_tr = torch.from_numpy(X_train[:n_tr].astype(np.float32)).to(device)
    y_tr = torch.from_numpy(y_train[:n_tr].astype(np.float32)).to(device)
    ds = torch.utils.data.TensorDataset(X_tr, y_tr)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    best_loss = float("inf")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(len(dl), 1)
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss

    return model, device, nfeat


def predict_lstm_torch(model, X: np.ndarray, device) -> np.ndarray:
    import torch
    model.eval()
    t = torch.from_numpy(X.astype(np.float32)).to(device)
    with torch.no_grad():
        return model(t).cpu().numpy()


def save_torch_lstm(model, path: Path, nfeat: int) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "nfeat": nfeat}, path)


def load_torch_lstm(path: Path):
    import torch
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    nfeat = int(ckpt["nfeat"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_lstm_torch_model(nfeat).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, device, nfeat


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def save_sklearn_model(model: Any, path: Path) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_sklearn_model(path: Path) -> Any:
    import joblib
    return joblib.load(path)


def save_keras_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_keras_model(path: Path):
    k = _keras()
    if k is None:
        raise RuntimeError("Keras not installed")
    keras, _ = k
    return keras.models.load_model(path)


def scale_fit(X_train: np.ndarray, X_test: np.ndarray | None = None):
    sc = StandardScaler()
    Xt = sc.fit_transform(X_train)
    if X_test is not None:
        Xv = sc.transform(X_test)
        return sc, Xt, Xv
    return sc, Xt
