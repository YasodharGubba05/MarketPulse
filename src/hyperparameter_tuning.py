"""Optuna hyperparameter tuning for XGBoost and LightGBM models.

Usage:
    from src.hyperparameter_tuning import tune_xgb_regressor, tune_lgbm_regressor
    best_params = tune_xgb_regressor(X_train, y_train, n_trials=30)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def tune_xgb_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 30,
    cv_folds: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Optuna search for XGBoost regressor hyperparams. Returns best_params dict."""
    try:
        import optuna
        from xgboost import XGBRegressor
        from sklearn.model_selection import cross_val_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "random_state": random_state,
                "n_jobs": -1,
            }
            model = XGBRegressor(**params)
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                                     scoring="neg_root_mean_squared_error")
            return -scores.mean()

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        logger.info("XGB tuning done. Best RMSE: %.4f | params: %s",
                    study.best_value, study.best_params)
        return study.best_params
    except ImportError as e:
        logger.warning("Optuna or XGBoost not available for tuning: %s. Using defaults.", e)
        return {}
    except Exception as e:
        logger.warning("XGB tuning failed: %s. Using defaults.", e)
        return {}


def tune_lgbm_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 30,
    cv_folds: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Optuna search for LightGBM regressor hyperparams."""
    try:
        import optuna
        import lightgbm as lgb
        from sklearn.model_selection import cross_val_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "random_state": random_state,
                "n_jobs": -1,
                "verbose": -1,
            }
            model = lgb.LGBMRegressor(**params)
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                                     scoring="neg_root_mean_squared_error")
            return -scores.mean()

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        logger.info("LGBM tuning done. Best RMSE: %.4f | params: %s",
                    study.best_value, study.best_params)
        return study.best_params
    except ImportError as e:
        logger.warning("Optuna or LightGBM not available for tuning: %s. Using defaults.", e)
        return {}
    except Exception as e:
        logger.warning("LGBM tuning failed: %s. Using defaults.", e)
        return {}


def tune_xgb_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 30,
    cv_folds: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Optuna search for XGBoost classifier hyperparams (crash detection)."""
    try:
        import optuna
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
                "random_state": random_state,
                "eval_metric": "logloss",
                "n_jobs": -1,
            }
            model = XGBClassifier(**params)
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                                     scoring="roc_auc")
            return -scores.mean()

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        logger.info("XGB Clf tuning done. Best ROC-AUC: %.4f | params: %s",
                    -study.best_value, study.best_params)
        return study.best_params
    except ImportError as e:
        logger.warning("Optuna or XGBoost not available for tuning: %s.", e)
        return {}
    except Exception as e:
        logger.warning("XGB Clf tuning failed: %s.", e)
        return {}
