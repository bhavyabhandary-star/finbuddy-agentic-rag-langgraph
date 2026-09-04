"""Train the Intent Router: TF-IDF + Logistic Regression, Ridge vs Lasso compared
across multiple train/test splits, every run logged to MLflow.

This is the project's own featured-deep-dive engineering discipline (build_prompt.md
Demo Narrative section) — applied to routing, not risk-trend prediction, since this
is the model the project actually creates and owns.

Run: python -m mlops.intent_router.train
"""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mlops.intent_router.data import load_training_examples

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
SPLITS_TO_TEST = [0.5, 0.2]  # test_size — mirrors the capstone's multi-split rigor
REGULARIZATIONS = {
    # liblinear doesn't support multiclass (we have 3 routes) — saga does.
    # sklearn >=1.8 deprecates the penalty="l1"/"l2" shorthand in favor of the
    # elasticnet l1_ratio parametrization (l1_ratio=0 -> pure L2/Ridge,
    # l1_ratio=1 -> pure L1/Lasso) — using penalty="l1" directly on this
    # version silently produced l1_ratio=0.0 (i.e. actually L2), which would
    # have made the two configs identical without erroring. Verified by
    # running this and reading the resulting FutureWarning, not assumed.
    "ridge_l2": dict(penalty="elasticnet", solver="saga", l1_ratio=0.0),
    "lasso_l1": dict(penalty="elasticnet", solver="saga", l1_ratio=1.0),
}


def _build_pipeline(**logreg_kwargs) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(min_df=1, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, **logreg_kwargs)),
        ]
    )


def _evaluate(pipeline: Pipeline, X_test: list[str], y_test: list[str]) -> dict:
    preds = pipeline.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision_macro": precision_score(y_test, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
    }


def train_and_log_all_configs() -> dict:
    """Trains every (regularization x split) combination, logs each to MLflow,
    returns the metrics dict for every run so the caller can pick a winner —
    same "full evaluation, every run" discipline as the capstone deep-dive.
    """
    texts, labels = load_training_examples()
    mlflow.set_experiment("finbuddy_langgraph_intent_router")

    results: dict[str, dict] = {}
    for reg_name, reg_kwargs in REGULARIZATIONS.items():
        for test_size in SPLITS_TO_TEST:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=42, stratify=labels
            )
            pipeline = _build_pipeline(**reg_kwargs)
            pipeline.fit(X_train, y_train)
            metrics = _evaluate(pipeline, X_test, y_test)

            run_name = f"{reg_name}_split_{int((1 - test_size) * 100)}_{int(test_size * 100)}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({"regularization": reg_name, "test_size": test_size, **reg_kwargs})
                mlflow.log_metrics(metrics)

            results[run_name] = {"metrics": metrics, "pipeline": pipeline}

    return results


def select_and_save_best(results: dict, metric: str = "f1_macro") -> str:
    """Picks the highest-scoring config on `metric` and saves it to artifacts/
    for mlops/intent_router/registry.py to promote. Returns the winning run name.

    NOTE: this seed dataset is tiny (milestone step 5's starter set) — this
    selection is meaningful once a real labeled dataset replaces it, not before.
    """
    import joblib

    best_run = max(results, key=lambda name: results[name]["metrics"][metric])
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(results[best_run]["pipeline"], ARTIFACTS_DIR / "intent_router.joblib")
    with open(ARTIFACTS_DIR / "intent_router_metrics.json", "w") as f:
        json.dump({"selected_run": best_run, **results[best_run]["metrics"]}, f, indent=2)
    return best_run


if __name__ == "__main__":
    all_results = train_and_log_all_configs()
    winner = select_and_save_best(all_results)
    print(f"Selected config: {winner}")
    for name, r in all_results.items():
        print(f"  {name}: {r['metrics']}")
