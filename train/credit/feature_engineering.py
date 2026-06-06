"""
Feature engineering for prime (customer snapshot) and transaction data.
Includes data cleaning steps (fill-na, outlier capping).
"""

import config
import numpy as np
import pandas as pd

_LEAKAGE_COLS = {
    "STATUS",
    "STATUS_NAME",
    "DELINQUENCY",
    "CARD_ACCOUNT_STATUS",
    "Card account status ",
    "CLOSURE_DATE",
}


# Helpers
def _get_numeric(df: pd.DataFrame, *col_names: str, default: float = 0.0) -> pd.Series:
    """Return the first matching column as a numeric Series, or a zero Series."""
    for col in col_names:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _assert_no_leakage(df: pd.DataFrame, stage: str) -> None:
    """Raise if any known leakage column survived into the feature DataFrame."""
    found = _LEAKAGE_COLS.intersection(df.columns)
    if found:
        raise ValueError(
            f"[leakage] The following columns must be dropped before '{stage}' "
            f"but are still present: {sorted(found)}\n"
            f"Add them to config.DROP_COLS and re-run."
        )


# Prime features
def engineer_prime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features from the prime (customer snapshot) data."""
    df = df.copy()
    extraction_date = pd.to_datetime("today")

    # --- Outlier capping (p99.9) for overdue amount ---
    # Bank stores overdue as negative (accounting convention) — flip to positive
    df["OVERDUEAMOUNT"] = df["OVERDUEAMOUNT"].abs()
    p99 = df["OVERDUEAMOUNT"].quantile(0.999)
    print(
        f"[feature_eng] Overdue outliers capped at {p99:.2f}: "
        f'{(df["OVERDUEAMOUNT"] > p99).sum()} rows'
    )
    df["OVERDUEAMOUNT"] = df["OVERDUEAMOUNT"].clip(upper=p99)

    # --- Core numeric references ---
    credit_limit = _get_numeric(df, "CREDIT_LIMIT")
    available_limit = _get_numeric(df, "AVAILABLE_LIMIT")
    ledger = _get_numeric(df, "LEDGER_BALANCE").abs()
    overdue = _get_numeric(df, "OVERDUEAMOUNT")

    credit_limit_safe = credit_limit.replace(0, np.nan)

    # --- Utilization ratios ---
    df["utilization_ratio"] = (ledger / credit_limit_safe).fillna(0)
    df["available_credit_ratio"] = (available_limit / credit_limit_safe).fillna(0)

    # --- Over-limit ratio ---
    over_limit_val = (ledger - credit_limit).clip(lower=0)
    df["over_limit_ratio"] = (over_limit_val / credit_limit_safe).fillna(0)

    # --- Overdue ratio ---
    df["overdue_ratio"] = (overdue / credit_limit_safe).fillna(0)

    # --- Composite financial stress score ---
    df["financial_stress_score"] = (
        df["utilization_ratio"] + df["over_limit_ratio"] * 2 + df["overdue_ratio"] * 3
    )

    # --- Fee-to-limit ratio ---
    annual_fee = _get_numeric(df, "ANNUAL_FEE")
    df["fee_to_limit_ratio"] = (annual_fee / credit_limit_safe).fillna(0)

    # --- Card replacement count ---
    replacement_cols = [
        "FIRST_REPLACED_CARD",
        "SECOND_REPLACED_CARD",
        "THIRD_REPLACED_CARD",
    ]
    existing_rep = [c for c in replacement_cols if c in df.columns]
    df["card_replacements"] = (
        df[existing_rep].notna().sum(axis=1) if existing_rep else 0
    )

    # --- Account tenure (months since creation) ---
    if "CREATION_DATE" in df.columns:
        df["account_tenure_months"] = (
            ((extraction_date - df["CREATION_DATE"]).dt.days / 30)
            .fillna(0)
            .clip(lower=0)
        )
    else:
        df["account_tenure_months"] = 0

    # --- Customer age ---
    if "DOB" in df.columns:
        df["customer_age"] = (
            ((extraction_date - df["DOB"]).dt.days / 365).fillna(0).clip(lower=0)
        )

    # Replace zero/missing ages with the median of valid ages
    median_age = df.loc[df["customer_age"] > 0, "customer_age"].median()

    df["customer_age"] = df["customer_age"].replace(0, median_age)

    print(f"[feature_eng] Prime features engineered -> {df.shape}")
    return df


# Transaction features
def engineer_transaction_features(txn_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level data into per-customer-per-month features.

    Returns a DataFrame indexed by ``(CUSTOMER_ID, snapshot_month)``.
    """
    df = txn_df.copy()
    cid = config.CUSTOMER_ID
    month_col = config.MONTH_COL
    extraction_date = pd.to_datetime("today")

    # Determine groupby keys: per customer per month if month tags exist
    group_keys = [cid, month_col]

    # --- Ensure numeric amounts ---
    for col in ["BILLING AMT", "ORIG AMOUNT", "SETTLEMENT AMT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- Outlier capping at p99.9 ---
    for col in ["BILLING AMT", "SETTLEMENT AMT", "ORIG AMOUNT"]:
        if col in df.columns:
            p99 = df[col].quantile(0.999)
            print(
                f"[feature_eng] {col} outliers capped at {p99:.2f}: "
                f"{(df[col] > p99).sum()} rows"
            )
            df[col] = df[col].clip(upper=p99)

    # --- Fill ORIG AMOUNT NaN with median ---
    if "ORIG AMOUNT" in df.columns:
        df["ORIG AMOUNT"] = df["ORIG AMOUNT"].fillna(df["ORIG AMOUNT"].median())

    # --- Transaction aggregations ---
    billing_col = "BILLING AMT"
    agg_dict: dict = {"txn_count": (cid, "size")}
    agg_dict.update(
        {
            "txn_total_amount": (billing_col, "sum"),
            "txn_mean_amount": (billing_col, "mean"),
            "txn_max_amount": (billing_col, "max"),
        }
    )

    agg = df.groupby(group_keys).agg(**agg_dict)

    # --- International transaction ratio ---
    df["_is_intl"] = (pd.to_numeric(df["CCY"], errors="coerce") != 818).astype(int)
    agg = agg.join(df.groupby(group_keys)["_is_intl"].sum().rename("txn_intl_count"))
    agg["txn_intl_ratio"] = (agg["txn_intl_count"] / agg["txn_count"]).where(
        agg["txn_count"] > 0, other=0
    )

    # --- Reversal ratio ---
    df["_is_reversal"] = (
        df["REVERSAL FLAG"]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(["Y", "YES", "1", "TRUE"])
    ).astype(int)
    agg = agg.join(
        df.groupby(group_keys)["_is_reversal"].sum().rename("txn_reversal_count")
    )
    agg["txn_reversal_ratio"] = (agg["txn_reversal_count"] / agg["txn_count"]).where(
        agg["txn_count"] > 0, other=0
    )

    # --- Merchant diversity ---
    agg = agg.join(
        df.groupby(group_keys)["MERCH ID"].nunique().rename("unique_merchants")
    )
    agg = agg.join(df.groupby(group_keys)["MCC"].nunique().rename("unique_mcc"))

    # --- Days since last transaction (recency) ---
    txn_recency = df.groupby(group_keys)["TRXN DATE"].max()
    agg["days_since_last_txn"] = (extraction_date - txn_recency).dt.days.fillna(-1)

    print(f"[feature_eng] Transaction features -> {agg.shape}")
    return agg


# Temporal / trend features
def _slope(series: pd.Series) -> float:
    """Compute the slope of a series over its positional index."""
    vals = series.dropna()
    if len(vals) < 2:
        return 0.0
    x = np.arange(len(vals))
    try:
        return float(np.polyfit(x, vals.values, 1)[0])
    except Exception:
        return 0.0


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trend features across months per customer, keep latest row.

    Takes the full picture of the customer across months
    (one row per customer per month, with engineered prime +
    transaction features and a ``snapshot_month`` column) and:

    1. Computes per-customer trends (slope, mean, std, delta) for the
       financial features across monthly snapshots.
    2. Adds aggregate counters (months_observed, months_with_overdue).
    3. Returns the latest month's snapshot, with all the trend features.
        (one row per customer)

    """
    cid = config.CUSTOMER_ID
    month_col = config.MONTH_COL

    if month_col not in df.columns:
        print("[feature_eng] No snapshot_month column — skipping temporal features.")
        return df

    df = df.sort_values([cid, month_col]).copy()

    unique_months = sorted(df[month_col].unique())
    n_months = len(unique_months)
    print(f"[feature_eng] Building temporal features across {n_months} month(s) ...")

    grouped = df.groupby(cid)

    # --- Aggregate counters ---
    trend_df = grouped[month_col].nunique().rename("months_observed").to_frame()

    if "overdue_ratio" in df.columns:
        trend_df["months_with_overdue"] = grouped["overdue_ratio"].apply(
            lambda s: (s > 0).sum()
        )

    # --- Trend / volatility for key feature columns ---
    all_trend_cols = [
        c for c in config.TREND_COLS + config.TXN_TREND_COLS if c in df.columns
    ]

    for col in all_trend_cols:
        col_grouped = grouped[col]
        trend_df[f"{col}_trend"] = col_grouped.apply(_slope)
        trend_df[f"{col}_mean"] = col_grouped.mean()
        trend_df[f"{col}_std"] = col_grouped.std().fillna(0)
        trend_df[f"{col}_delta"] = col_grouped.apply(
            lambda s: s.iloc[-1] - s.iloc[0] if len(s) >= 2 else 0.0
        )

    # --- Keep only the latest row per customer ---
    latest_idx = grouped[month_col].idxmax()
    latest_df = df.loc[latest_idx].copy()

    # Merge trend features onto the latest row
    latest_df = latest_df.merge(
        trend_df,
        left_on=cid,
        right_index=True,
        how="left",
    )

    latest_df = latest_df.reset_index(drop=True)

    print(f"[feature_eng] Temporal features -> {latest_df.shape}")
    print(
        f"  {n_months} months, {len(trend_df)} unique customers, "
        f"{len(all_trend_cols) * 4 + 2} trend features added"
    )
    return latest_df


# Target creation
def create_target(df: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
    """Create the target column.

    Modes
    -----
    hard_only       : 1 for 60DA / 90DA / WROF only
    weighted_binary : 1 for all defaults; soft defaults get weight = 0.4
    tiered          : 0 = normal, 1 = soft default, 2 = hard default
    all             : 1 for all DEFAULT_STATUSES
    """
    if mode is None:
        mode = getattr(config, "TARGET_MODE", "all")

    df = df.copy()

    if mode == "hard_only":
        df["target"] = (
            df[config.STATUS_COL].isin(config.HARD_DEFAULT_STATUSES).astype(int)
        )

    elif mode == "weighted_binary":
        all_defaults = config.HARD_DEFAULT_STATUSES + config.SOFT_DEFAULT_STATUSES
        df["target"] = df[config.STATUS_COL].isin(all_defaults).astype(int)
        weight_map = {
            **{s: 1.0 for s in config.HARD_DEFAULT_STATUSES},
            **{s: 0.4 for s in config.SOFT_DEFAULT_STATUSES},
        }
        df["sample_weight"] = df[config.STATUS_COL].map(weight_map).fillna(0.0)

    elif mode == "tiered":
        df["target"] = np.where(
            df[config.STATUS_COL].isin(config.HARD_DEFAULT_STATUSES),
            2,
            np.where(df[config.STATUS_COL].isin(config.SOFT_DEFAULT_STATUSES), 1, 0),
        )

    elif mode == "all":
        df["target"] = df[config.STATUS_COL].isin(config.DEFAULT_STATUSES).astype(int)

    else:
        raise ValueError(
            f"Unknown TARGET_MODE '{mode}'. "
            "Choose from: hard_only, weighted_binary, tiered, all."
        )

    return df
