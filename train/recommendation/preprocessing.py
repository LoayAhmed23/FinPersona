import config
import pandas as pd


def fill_missing_values(profile):
    mcc_cols = [
        col
        for col in profile.columns
        if col.startswith("MCC_") and col.endswith("_SPEND")
    ]
    fill_zero_cols = [
        "TOTAL_SPEND_AMT",
        "AVG_TRXN_AMT",
        "TRXN_COUNT",
        "FOREIGN_TRXN_COUNT",
    ] + mcc_cols
    for c in fill_zero_cols:
        if c in profile.columns:
            profile[c] = profile[c].fillna(0)
    if "DAYS_SINCE_LAST_TRXN" in profile.columns:
        profile["DAYS_SINCE_LAST_TRXN"] = profile["DAYS_SINCE_LAST_TRXN"].fillna(9999)
    return profile


def one_hot_encode_categoricals(profile):
    if "AGE_GROUP" in profile.columns:
        age_group_dummies = pd.get_dummies(
            profile["AGE_GROUP"], prefix="AGE_GROUP", drop_first=False
        )
        age_group_dummies.columns = [
            col.replace("-", "_") for col in age_group_dummies.columns
        ]
        profile = pd.concat([profile, age_group_dummies], axis=1)
        profile = profile.drop(columns=["AGE_GROUP"])

    if "GENDER" in profile.columns:
        gender_dummies = pd.get_dummies(
            profile["GENDER"], prefix="GENDER", drop_first=False
        )
        profile = pd.concat([profile, gender_dummies], axis=1)
        profile = profile.drop(columns=["GENDER"])

    if "BRANCH_ID" in profile.columns:
        branch_id_dummies = pd.get_dummies(
            profile["BRANCH_ID"], prefix="BRANCH_ID", drop_first=False
        )
        profile = pd.concat([profile, branch_id_dummies], axis=1)
        profile = profile.drop(columns=["BRANCH_ID"])

    return profile


def drop_unneeded_columns(profile, prime_only=False, txn_only=False, final_stage=False):
    if prime_only:
        existing_cols_to_drop = [
            col for col in config.PRIME_COLUMNS_TO_DROP if col in profile.columns
        ]
        profile = profile.drop(columns=existing_cols_to_drop)
    if txn_only:
        existing_cols_to_drop = [
            col for col in config.TRANSACTION_COLUMNS_TO_DROP if col in profile.columns
        ]
        profile = profile.drop(columns=existing_cols_to_drop)
    if final_stage:
        final_columns_to_drop = [
            "AGE",
            "BRANCH_NAME",
            "PRODUCT_NAME",
            "DOB",
            "RIMNO",
            "DOB_WAS_MISSING",
            "GENDER_Unknown",
        ]
        existing_cols_to_drop = [
            col for col in final_columns_to_drop if col in profile.columns
        ]
        profile = profile.drop(columns=existing_cols_to_drop)
        profile = profile.drop(columns=["BRANCH_ID"], errors="ignore")
    return profile


def filter_uncorrelated_features(profile, logs):
    product_cols = [col for col in profile.columns if col.startswith("HAS_PROD_")]
    exclude_cols_corr = ["CUSTOMER_ID"] + product_cols
    feature_cols = [col for col in profile.columns if col not in exclude_cols_corr]

    if not product_cols:
        return profile, feature_cols, product_cols

    logs.append(
        f"Analyzing {len(feature_cols)} features against {len(product_cols)} products..."
    )

    corr_matrix = profile[feature_cols + product_cols].corr()
    feature_product_corr = corr_matrix.loc[feature_cols, product_cols]
    max_corr_per_feature = feature_product_corr.abs().max(axis=1)

    features_to_keep = max_corr_per_feature[
        max_corr_per_feature >= config.CORR_THRESHOLD
    ].index.tolist()
    features_to_drop = max_corr_per_feature[
        max_corr_per_feature < config.CORR_THRESHOLD
    ].index.tolist()

    logs.append(f"Keeping {len(features_to_keep)} strongly correlated features.")
    logs.append(f"Dropping {len(features_to_drop)} weak/uncorrelated features.")

    profile = profile.drop(columns=features_to_drop)
    return profile, features_to_keep, product_cols


def drop_low_volume_products(profile, logs):
    current_product_cols = [
        col for col in profile.columns if col.startswith("HAS_PROD_")
    ]
    if not current_product_cols:
        return profile

    product_counts = profile[current_product_cols].sum()
    products_to_drop = product_counts[product_counts < 100].index.tolist()
    logs.append(
        f"Dropping {len(products_to_drop)} products with fewer than 100 holders."
    )
    if products_to_drop:
        profile = profile.drop(columns=products_to_drop)
    return profile


def preprocess_pipeline(profile, logs):
    profile = fill_missing_values(profile)
    profile = one_hot_encode_categoricals(profile)
    profile = drop_unneeded_columns(
        profile, prime_only=True, txn_only=True, final_stage=True
    )
    profile, _, _ = filter_uncorrelated_features(profile, logs)
    profile = drop_low_volume_products(profile, logs)

    final_products = [col for col in profile.columns if col.startswith("HAS_PROD_")]
    final_features = [
        col
        for col in profile.columns
        if col not in final_products and col != "CUSTOMER_ID"
    ]

    logs.append("")
    logs.append("=" * 60)
    logs.append(" FINAL DATASET SUMMARY")
    logs.append("=" * 60)
    logs.append(f"Total Customers:         {len(profile)}")
    logs.append(f"Total Features Retained: {len(final_features)}")
    logs.append(f"Total Products Retained: {len(final_products)}")

    logs.append("")
    logs.append("Extracted Features:")
    for i, feat in enumerate(final_features, 1):
        logs.append(f"  {i:3d}. {feat}")

    logs.append("")
    logs.append("Target Products:")
    for i, prod in enumerate(final_products, 1):
        logs.append(f"  {i:3d}. {prod.replace('HAS_PROD_', '')}")

    return profile, final_products, final_features
