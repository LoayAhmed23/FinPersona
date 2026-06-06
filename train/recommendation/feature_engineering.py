import pandas as pd


def build_user_item_matrix(prime_df, logs):
    user_item_df = None
    if "PRODUCT_NAME" in prime_df.columns and prime_df["PRODUCT_NAME"].notna().any():
        user_item_matrix = pd.crosstab(
            prime_df["CUSTOMER_ID"], prime_df["PRODUCT_NAME"]
        )
        user_item_matrix.columns = [
            f"HAS_PROD_{col.strip().replace(' ', '_')}"
            for col in user_item_matrix.columns
        ]
        user_item_matrix = (user_item_matrix > 0).astype(int)
        user_item_df = user_item_matrix.copy()
        prime_df = prime_df.drop_duplicates(subset=["CUSTOMER_ID"]).merge(
            user_item_matrix, on="CUSTOMER_ID", how="inner"
        )
        logs.append(
            f"  Built user-item matrix: {len(user_item_matrix.columns)} products detected."
        )
    else:
        prime_df = prime_df.drop_duplicates(subset=["CUSTOMER_ID"])
        logs.append("  No PRODUCT_NAME found — skipping user-item matrix.")
    return prime_df, user_item_df


def build_rfm_features(transaction_df):  # Extracts Recency, Frequency, Monetary features from transactions.
    if transaction_df is None or len(transaction_df) == 0:
        return pd.DataFrame(
            columns=[
                "CUSTOMER_ID",
                "TOTAL_SPEND_AMT",
                "AVG_TRXN_AMT",
                "TRXN_COUNT",
                "DAYS_SINCE_LAST_TRXN",
            ]
        )

    rfm_features = (
        transaction_df.groupby("CUSTOMER_ID")
        .agg(
            TOTAL_SPEND_AMT=("BILLING AMT", "sum"),
            AVG_TRXN_AMT=("BILLING AMT", "mean"),
            TRXN_COUNT=("BILLING AMT", "count"),
            DAYS_SINCE_LAST_TRXN=(
                "TRXN DATE",
                lambda x: (pd.to_datetime("today") - x.max()).days,
            ),
        )
        .reset_index()
    )
    return rfm_features


def build_mcc_spend(transaction_df):
    if transaction_df is None or len(transaction_df) == 0:
        return pd.DataFrame(columns=["CUSTOMER_ID"])

    mcc_spend = pd.pivot_table(
        transaction_df,
        values="BILLING AMT",
        index="CUSTOMER_ID",
        columns="MCC",
        aggfunc="sum",
        fill_value=0,
    )
    mcc_spend.columns = [f"MCC_{str(col)}_SPEND" for col in mcc_spend.columns]
    mcc_spend = mcc_spend.reset_index()
    return mcc_spend


def build_foreign_trxn_features(transaction_df):
    if transaction_df is None or len(transaction_df) == 0:
        return pd.DataFrame(columns=["CUSTOMER_ID", "FOREIGN_TRXN_COUNT"])

    transaction_df["IS_FOREIGN_TRXN"] = (
        (transaction_df["TRXN COUNTRY"] != "EGYPT").fillna(False).astype(int)
    )
    foreign_agg = (
        transaction_df.groupby("CUSTOMER_ID")
        .agg(FOREIGN_TRXN_COUNT=("IS_FOREIGN_TRXN", "sum"))
        .reset_index()
    )
    return foreign_agg


def build_demographics_features(prime_df):
    extraction_date = pd.to_datetime("today")
    if "DOB" in prime_df.columns:
        prime_df["AGE"] = (extraction_date - prime_df["DOB"]).dt.days // 365
        bins = [18, 25, 35, 50, 65, 100]
        labels = ["18-25", "26-35", "36-50", "51-65", "65+"]
        prime_df["AGE_GROUP"] = pd.cut(
            prime_df["AGE"], bins=bins, labels=labels, right=True
        )
    else:
        prime_df["AGE"] = pd.NA
        prime_df["AGE_GROUP"] = pd.NA
    return prime_df


def merge_all_features(prime_df, rfm_features, mcc_spend, foreign_agg, logs):
    logs.append("")
    logs.append("=" * 60)
    logs.append(" PHASE 3: Building Final Customer Profile (1 Row per Customer)")
    logs.append("=" * 60)

    profile = prime_df.copy()
    profile = profile.merge(rfm_features, on="CUSTOMER_ID", how="left")
    profile = profile.merge(mcc_spend, on="CUSTOMER_ID", how="left")
    profile = profile.merge(foreign_agg, on="CUSTOMER_ID", how="left")

    return profile
