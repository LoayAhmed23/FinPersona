import pandas as pd
from conftest import load_project_module


def test_prime_cleaning_casts_numeric_dates_and_strings():
    prime_cleaning = load_project_module(
        "data_cleaning/prime_id_creation.py",
        "prime_cleaning_test",
    )

    df = pd.DataFrame(
        {
            "RIMNO": ["1,001", "bad"],
            "CREDIT_LIMIT": ["1,500.50", "oops"],
            "CREATION_DATE": ["2026-02-01", "not a date"],
            "GENDER": ["F", None],
        }
    )

    prime_cleaning.apply_cast_and_report(df, ["RIMNO"], "int")
    prime_cleaning.apply_cast_and_report(df, ["CREDIT_LIMIT"], "float")
    prime_cleaning.apply_cast_and_report(df, ["CREATION_DATE"], "date")
    prime_cleaning.apply_cast_and_report(df, ["GENDER"], "string")

    assert str(df["RIMNO"].dtype) == "Int64"
    assert df["RIMNO"].tolist()[0] == 1001
    assert pd.isna(df["RIMNO"].tolist()[1])
    assert df["CREDIT_LIMIT"].tolist()[0] == 1500.50
    assert pd.isna(df["CREDIT_LIMIT"].tolist()[1])
    assert df["CREATION_DATE"].tolist()[0] == pd.Timestamp("2026-02-01")
    assert pd.isna(df["CREATION_DATE"].tolist()[1])
    assert str(df["GENDER"].dtype) == "string"


def test_prime_cleaning_drops_rows_missing_critical_columns():
    prime_cleaning = load_project_module(
        "data_cleaning/prime_id_creation.py",
        "prime_drop_empty_test",
    )

    df = pd.DataFrame(
        {
            "RIMNO": ["1001", None, "1003"],
            "PRODUCT_NAME": ["Card", "Card", None],
        }
    )

    cleaned = prime_cleaning.drop_empty_records(df, ["RIMNO", "PRODUCT_NAME"])

    assert cleaned.to_dict("records") == [
        {"RIMNO": "1001", "PRODUCT_NAME": "Card"},
    ]


def test_prime_cleaning_missing_value_summary_reports_only_missing_columns():
    prime_cleaning = load_project_module(
        "data_cleaning/prime_id_creation.py",
        "prime_missing_summary_test",
    )

    df = pd.DataFrame(
        {
            "RIMNO": ["1001", "1002", "1003", "1004"],
            "GENDER": ["F", None, None, "M"],
            "BRANCH_NAME": ["A", "B", "C", "D"],
        }
    )

    summary = prime_cleaning.check_missing_values(df)

    assert summary.index.tolist() == ["GENDER"]
    assert summary.loc["GENDER", "Missing Count"] == 2
    assert summary.loc["GENDER", "Percentage (%)"] == 50.0


def test_transaction_cleaning_cast_helper_normalizes_transaction_columns():
    transaction_cleaning = load_project_module(
        "data_cleaning/transaction_id_mapping.py",
        "transaction_cleaning_test",
    )

    df = pd.DataFrame(
        {
            "BILLING AMT": ["2,000.25", "bad"],
            "MCC": ["5411", "x"],
            "TRXN DATE": ["2026-03-15", "bad date"],
        }
    )

    transaction_cleaning.apply_cast_and_report(df, ["BILLING AMT"], "float")
    transaction_cleaning.apply_cast_and_report(df, ["MCC"], "int")
    transaction_cleaning.apply_cast_and_report(df, ["TRXN DATE"], "date")

    assert df["BILLING AMT"].tolist()[0] == 2000.25
    assert pd.isna(df["BILLING AMT"].tolist()[1])
    assert str(df["MCC"].dtype) == "Int64"
    assert df["MCC"].tolist()[0] == 5411
    assert pd.isna(df["MCC"].tolist()[1])
    assert df["TRXN DATE"].tolist()[0] == pd.Timestamp("2026-03-15")
