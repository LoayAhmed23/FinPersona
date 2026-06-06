import config
import numpy as np
import pandas as pd
from evaluation import (
    evaluate_cbf_metrics,
    evaluate_xgboost_metrics,
    optimize_cbf_thresholds,
    optimize_xgboost_thresholds,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def train_xgboost(df, logs=None):
    if logs is None:
        logs = []
    logs.append("=" * 60)
    logs.append(" XGBoost + Per-Product Threshold Tuning")
    logs.append("=" * 60)

    target_cols = [col for col in df.columns if col.startswith("HAS_PROD_")]
    exclude_cols = ["CUSTOMER_ID", "BRANCH_ID"] + target_cols
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].fillna(0)

    valid_targets = [
        col for col in target_cols if df[col].sum() >= config.MIN_SAMPLES_REQUIRED
    ]

    logs.append(f"Total Features:  {len(feature_cols)}")
    logs.append(f"Valid Products:  {len(valid_targets)}")

    Y = df[valid_targets]

    X_train_full, X_test, Y_train_full, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_full, Y_train_full, test_size=0.2, random_state=42
    )

    logs.append(f"Train size:      {len(X_train)}")
    logs.append(f"Validation size: {len(X_val)}")
    logs.append(f"Test size:       {len(X_test)}")

    # Save test set customer IDs to a text file
    test_customer_ids = df.loc[X_test.index, "CUSTOMER_ID"].astype(int).tolist()
    with open("test_customer_ids.txt", "w") as f:
        for cid in test_customer_ids:
            f.write(f"{cid}\n")
    logs.append(
        f"Saved {len(test_customer_ids)} test customer IDs to test_customer_ids.txt"
    )

    # Train one XGBoost per product
    val_proba_dict = {}
    test_proba_dict = {}
    models_dict = {}

    logs.append("")
    logs.append("Training XGBoost models (one per product)...")

    for i, product in enumerate(valid_targets):
        y_train_col = Y_train[product].values
        positives = y_train_col.sum()
        negatives = len(y_train_col) - positives
        spw = negatives / (positives + 1e-5)

        xgb_model = XGBClassifier(
            scale_pos_weight=spw,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr",
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
            n_jobs=-1,
        )
        xgb_model.fit(X_train, y_train_col)
        models_dict[product] = xgb_model

        val_proba_dict[product] = xgb_model.predict_proba(X_val)[:, 1]
        test_proba_dict[product] = xgb_model.predict_proba(X_test)[:, 1]

        if (i + 1) % 10 == 0 or (i + 1) == len(valid_targets):
            logs.append(
                f"  Trained {i+1}/{len(valid_targets)} models "
                f"(last: {product.replace('HAS_PROD_', '')}  spw={spw:.1f})"
            )

    logs.append("All models trained.")

    # Threshold optimization
    logs.append("")
    logs.append("Optimizing decision thresholds on validation set...")
    optimal_thresholds = optimize_xgboost_thresholds(
        valid_targets, val_proba_dict, Y_val
    )

    # Apply on test set
    predictions_dict = {}
    for product in valid_targets:
        t = optimal_thresholds[product]
        predictions_dict[product] = (test_proba_dict[product] >= t).astype(int)

    Y_pred = pd.DataFrame(predictions_dict, index=Y_test.index)

    # measuring metrics
    metrics, exact_acc, micro_f1, macro_f1 = evaluate_xgboost_metrics(
        Y_test, Y_pred, valid_targets, Y_train, optimal_thresholds
    )

    logs.append("")
    logs.append(f"Exact Match Accuracy: {exact_acc * 100:.2f}%")
    logs.append(f"Micro F1: {micro_f1:.4f} | Macro F1: {macro_f1:.4f}")

    return models_dict, optimal_thresholds, feature_cols, valid_targets, metrics, logs


def predict_for_customer(
    customer_id, df, models_dict, optimal_thresholds, feature_cols, valid_targets
):
    customer_row = df[df["CUSTOMER_ID"] == customer_id]
    if customer_row.empty:
        return [], [], False

    X_customer = customer_row[feature_cols].fillna(0)

    already_holding = []
    for product in valid_targets:
        if product in customer_row.columns:
            if int(customer_row[product].values[0]) == 1:
                already_holding.append(product.replace("HAS_PROD_", ""))

    # Model predictions
    model_predictions = []  # all products the model predicts (above threshold)

    for product in valid_targets:
        model = models_dict[product]
        proba = model.predict_proba(X_customer)[:, 1][0]
        threshold = optimal_thresholds[product]
        product_name = product.replace("HAS_PROD_", "")
        currently_holds = product_name in already_holding

        if proba >= threshold:
            model_predictions.append(
                {
                    "product": product_name,
                    "probability": round(float(proba) * 100, 2),
                    "threshold": threshold,
                    "currently_holds": currently_holds,
                }
            )
    return model_predictions, already_holding, True


def train_cbf(df, logs=None):
    if logs is None:
        logs = []
    logs.append("=" * 60)
    logs.append(" Content-Based Filtering (Item-Item Similarity)")
    logs.append("=" * 60)

    product_cols = [col for col in df.columns if col.startswith("HAS_PROD_")]
    user_item = df.set_index("CUSTOMER_ID")[product_cols]
    user_item = (user_item > 0).astype(int)

    logs.append(f"Users:    {user_item.shape[0]}")
    logs.append(f"Products: {user_item.shape[1]}")

    # Masking: train/test split
    np.random.seed(42)
    train_matrix = user_item.copy()
    test_matrix = pd.DataFrame(0, index=user_item.index, columns=user_item.columns)

    users_with_test = 0
    for idx in range(user_item.shape[0]):
        held = np.where(user_item.iloc[idx].values > 0)[0]
        if len(held) > 1:
            n_mask = max(1, int(len(held) * 0.2))
            masked = np.random.choice(held, size=n_mask, replace=False)
            test_matrix.iloc[idx, masked] = 1
            train_matrix.iloc[idx, masked] = 0
            users_with_test += 1

    logs.append(f"Train/Test masking: {users_with_test} users have masked items.")

    # Build similarity on TRAIN data only
    sim_array = cosine_similarity(train_matrix.T)
    sim_matrix = pd.DataFrame(sim_array, index=product_cols, columns=product_cols)

    # Compute score matrix (users × products) via matrix multiply
    train_np = train_matrix.values  # (n_users, n_products)
    sim_np = sim_matrix.values  # (n_products, n_products)
    score_matrix = train_np @ sim_np  # (n_users, n_products)

    # Mask: set score to -inf for products the user already holds
    held_mask = train_np > 0
    score_matrix_masked = score_matrix.copy()
    score_matrix_masked[held_mask] = -np.inf

    # Identify users who have test items
    test_np = test_matrix.values
    test_user_mask = test_np.sum(axis=1) > 0  # (n_users,) bool

    logs.append(f"Evaluable users: {test_user_mask.sum()}")

    if test_user_mask.sum() == 0:
        logs.append(
            "WARNING: No users have multiple products to mask. Skipping CBF threshold optimization and evaluation."
        )
        cbf_thresholds = {col: 0.1 for col in product_cols}
        metrics = {}
    else:
        logs.append("")
        logs.append("Optimizing similarity thresholds (recall-focused F2-score)...")
        cbf_thresholds = optimize_cbf_thresholds(
            product_cols, score_matrix_masked, test_user_mask, test_np
        )
        logs.append("Thresholds optimized (recall-weighted).")

        logs.append("")
        logs.append("Evaluating on test set...")
        metrics, exact_acc, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1 = (
            evaluate_cbf_metrics(
                score_matrix_masked,
                test_np,
                test_user_mask,
                cbf_thresholds,
                product_cols,
            )
        )

        logs.append(f"Exact Match Accuracy: {exact_acc * 100:.2f}%")
        logs.append(f"Micro  — P: {micro_p:.4f}  R: {micro_r:.4f}  F1: {micro_f1:.4f}")
        logs.append(f"Macro  — P: {macro_p:.4f}  R: {macro_r:.4f}  F1: {macro_f1:.4f}")

    full_sim_array = cosine_similarity(user_item.T)
    sim_matrix_full = pd.DataFrame(
        full_sim_array, index=product_cols, columns=product_cols
    )

    return sim_matrix_full, product_cols, cbf_thresholds, metrics, logs


def predict_cbf_for_customer(customer_id, df, sim_matrix, product_cols, cbf_thresholds):
    customer_row = df[df["CUSTOMER_ID"] == customer_id]
    if customer_row.empty:
        return [], [], False

    held_cols = []
    for col in product_cols:
        if col in customer_row.columns:
            val = customer_row[col].values[0]
            if pd.notna(val) and float(val) == 1:
                held_cols.append(col)
    already_holding = [col.replace("HAS_PROD_", "") for col in held_cols]

    if not held_cols:
        return [], already_holding, True

    recommendations = []
    for product in product_cols:
        if product in held_cols:
            continue
        score = sim_matrix.loc[product, held_cols].sum()
        threshold = cbf_thresholds.get(product, 0.5)
        if score >= threshold:
            recommendations.append(
                {
                    "product": product.replace("HAS_PROD_", ""),
                    "similarity_score": round(float(score), 4),
                    "threshold": threshold,
                }
            )

    recommendations.sort(key=lambda x: x["similarity_score"], reverse=True)

    return recommendations, already_holding, True
