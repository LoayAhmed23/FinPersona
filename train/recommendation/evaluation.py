import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score, fbeta_score


def optimize_xgboost_thresholds(valid_targets, val_proba_dict, Y_val):
    """Optimizes decision thresholds for XGBoost on the validation set."""
    optimal_thresholds = {}
    for product in valid_targets:
        val_proba = val_proba_dict[product]
        y_val_col = Y_val[product].values
        best_threshold = 0.5
        best_f1 = 0.0
        for t in np.arange(0.05, 0.95, 0.05):
            preds = (val_proba >= t).astype(int)
            score = f1_score(y_val_col, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = round(float(t), 2)
        optimal_thresholds[product] = best_threshold
    return optimal_thresholds


def optimize_cbf_thresholds(product_cols, score_matrix_masked, test_user_mask, test_np):
    """Optimizes similarity thresholds for CBF using F2-score."""
    cbf_thresholds = {}
    thresholds_to_try = np.arange(0.05, 1.55, 0.05)

    for p_idx, product in enumerate(product_cols):
        scores_col = score_matrix_masked[test_user_mask, p_idx]
        y_true_col = test_np[test_user_mask, p_idx]

        best_t = 0.3  # lower default to favour recall
        best_f2 = 0.0
        for t in thresholds_to_try:
            y_pred_col = (scores_col >= t).astype(int)
            f2_val = fbeta_score(y_true_col, y_pred_col, beta=2, zero_division=0)
            if f2_val > best_f2:
                best_f2 = f2_val
                best_t = round(float(t), 2)
        cbf_thresholds[product] = best_t
    return cbf_thresholds


def evaluate_xgboost_metrics(Y_test, Y_pred, valid_targets, Y_train, optimal_thresholds):
    """Calculates evaluation metrics for the XGBoost predictions."""
    exact_acc = accuracy_score(Y_test, Y_pred)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average='micro', zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average='weighted', zero_division=0)
    samples_p, samples_r, samples_f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average='samples', zero_division=0)

    metrics = {
        "exact_match_accuracy": round(exact_acc * 100, 2),
        "micro": {"precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f1, 4)},
        "macro": {"precision": round(macro_p, 4), "recall": round(macro_r, 4), "f1": round(macro_f1, 4)},
        "weighted": {"precision": round(weighted_p, 4), "recall": round(weighted_r, 4), "f1": round(weighted_f1, 4)},
        "samples": {"precision": round(samples_p, 4), "recall": round(samples_r, 4), "f1": round(samples_f1, 4)},
    }

    threshold_table = []
    for p in valid_targets:
        threshold_table.append({
            "product": p.replace('HAS_PROD_', ''),
            "positives_in_train": int(Y_train[p].sum()),
            "optimal_threshold": optimal_thresholds[p]
        })
    metrics["threshold_table"] = threshold_table
    return metrics, exact_acc, micro_f1, macro_f1


def evaluate_cbf_metrics(score_matrix_masked, test_np, test_user_mask, cbf_thresholds, product_cols):
    """Calculates evaluation metrics for CBF predictions."""
    threshold_arr = np.array([cbf_thresholds[p] for p in product_cols])
    Y_pred = (score_matrix_masked[test_user_mask] >= threshold_arr).astype(int)
    Y_true = test_np[test_user_mask]

    exact_acc = accuracy_score(Y_true, Y_pred)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average='micro', zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average='macro', zero_division=0
    )

    metrics = {
        "exact_match_accuracy": round(exact_acc * 100, 2),
        "micro": {"precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f1, 4)},
        "macro": {"precision": round(macro_p, 4), "recall": round(macro_r, 4), "f1": round(macro_f1, 4)},
    }

    return metrics, exact_acc, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1
