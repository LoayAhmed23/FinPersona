"""
Model training, hyperparameter tuning, and saving the best trained model.
"""

import config
import joblib
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# Model definitions
def get_classifiers(y_train=None):
    """Return a dict of named classifiers with sensible defaults."""
    # Compute scale_pos_weight for XGBoost
    spw = 1.0
    if y_train is not None:
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw = n_neg / n_pos if n_pos > 0 else 1.0

    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            max_iter=1000,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }


# Training
def train_classifiers(X_train, y_train, X_test, y_test) -> dict:
    """Train all classifiers and return predictions.

    Returns dict: {name: {"model": fitted_model, "preds": array, "probs": array}}
    """
    classifiers = get_classifiers(y_train)
    results = {}

    for name, clf in classifiers.items():
        print(f"  Training {name} ...")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
        results[name] = {"model": clf, "preds": preds, "probs": probs}
        print("    Done.")

    return results


# Hyperparameter tuning (GridSearch)
def tune_models(X_train, y_train) -> dict:
    """Run GridSearchCV for Random Forest and XGBoost.

    Returns dict: {name: {"model": best_estimator, "best_params": dict}}
    """
    skf = StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    spw = 1.0
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    if n_pos > 0:
        spw = n_neg / n_pos

    tuning_jobs = {
        "Random Forest": {
            "estimator": RandomForestClassifier(
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            ),
            "params": config.RF_GRID_PARAMS,
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                eval_metric="logloss",
                scale_pos_weight=spw,
                random_state=config.RANDOM_STATE,
            ),
            "params": config.XGB_GRID_PARAMS,
        },
    }

    results = {}
    for name, job in tuning_jobs.items():
        print(f"  Tuning {name} ...")
        grid = GridSearchCV(
            estimator=job["estimator"],
            param_grid=job["params"],
            scoring=config.CV_SCORING,
            cv=skf,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        print(f"    Best {config.CV_SCORING}: {grid.best_score_:.4f}")
        print(f"    Best params: {grid.best_params_}")
        results[name] = {
            "model": grid.best_estimator_,
            "best_params": grid.best_params_,
        }

    return results


# Saving the model
def save_model(model, artifacts: dict, path: str = None):
    """Persist model and preprocessing artifacts."""
    import os

    path = path or config.MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model": model, "artifacts": artifacts}
    joblib.dump(payload, path)
    print(f"[model] Saved to {path}")


def load_model(path: str = None):
    """Load model and preprocessing artifacts."""
    path = path or config.MODEL_PATH
    payload = joblib.load(path)
    print(f"[model] Loaded from {path}")
    return payload["model"], payload["artifacts"]
