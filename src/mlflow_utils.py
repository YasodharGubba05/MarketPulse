"""Optional MLflow tracking (file store under data/mlruns)."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

def _default_mlflow_uri() -> str:
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    p = root / "data" / "mlruns"
    return "file:" + str(p)


_MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", _default_mlflow_uri())


@contextmanager
def mlflow_run(
    experiment_name: str = "multi-stock-ads",
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    """Start an MLflow run if mlflow is installed; otherwise yield None once."""
    try:
        import mlflow
    except ImportError:
        logger.info("MLflow not installed; skipping experiment tracking. pip install mlflow")
        yield None
        return

    mlflow.set_tracking_uri(_MLFLOW_URI)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, v)
        yield run


def log_params_safe(**kwargs: Any) -> None:
    try:
        import mlflow

        flat = {k: (str(v) if not isinstance(v, (int, float, bool)) else v) for k, v in kwargs.items()}
        mlflow.log_params(flat)
    except Exception as e:
        logger.debug("mlflow log_params: %s", e)


def log_metrics_safe(prefix: str, metrics: dict[str, Any]) -> None:
    try:
        import mlflow

        for k, v in metrics.items():
            if isinstance(v, (int, float)) and v == v:  # not nan
                mlflow.log_metric(f"{prefix}/{k}", float(v))
    except Exception as e:
        logger.debug("mlflow log_metrics: %s", e)


def log_nested_json_artifact(obj: dict[str, Any], filename: str = "metrics_snapshot.json") -> None:
    """Save metrics blob as MLflow artifact (active run required)."""
    import json
    import tempfile

    try:
        import mlflow

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(obj, f, indent=2, default=str)
            path = f.name
        mlflow.log_artifact(path, artifact_path="metrics")
        try:
            os.unlink(path)
        except OSError:
            pass
    except Exception as e:
        logger.debug("mlflow artifact: %s", e)


def log_metric_flat(prefix: str, data: dict[str, Any], max_depth: int = 3) -> None:
    """Flatten nested dicts slightly for MLflow metrics tab."""

    def walk(d: dict[str, Any], pfx: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            import mlflow
        except ImportError:
            return
        for k, v in d.items():
            key = f"{pfx}/{k}" if pfx else k
            if isinstance(v, dict):
                walk(v, key, depth + 1)
            elif isinstance(v, (int, float)) and v == v:
                try:
                    mlflow.log_metric(key.replace(" ", "_")[:250], float(v))
                except Exception:
                    pass

    walk(data, prefix, 0)
