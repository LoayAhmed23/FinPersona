"""
Generate fake Prime and Transaction data for local testing.

The generated dataset is designed to support all three model families:

- credit risk: monthly utilization, overdue amount, and delinquency/status drift
- churn: customers that disappear by MAY plus a small number of write-offs
- recommendation: multiple correlated card products per customer

Outputs
-------
data/prime/FEB2026.csv, MAR2026.csv, APR2026.csv, MAY2026.csv
data/transaction/202602.xlsx, 202603.xlsx, 202604.xlsx, 202605.xlsx

Run:  python data/generate_fake_data.py
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRIME_OUT = os.path.join(SCRIPT_DIR, "prime")
TXN_OUT = os.path.join(SCRIPT_DIR, "transaction")
os.makedirs(PRIME_OUT, exist_ok=True)
os.makedirs(TXN_OUT, exist_ok=True)

N_BASE_CUSTOMERS = 1000
N_NEW_MAY = 80
N_DROP_MAY = 60

BRANCHES = [
    (1, "Cairo Main"), (2, "Giza"), (3, "Alexandria"),
    (4, "Mansoura"), (5, "Tanta"), (6, "Assiut"),
    (7, "Luxor"), (8, "Zagazig"), (9, "Ismailia"), (10, "Suez"),
]

PRODUCT_NAMES = [
    "Classic Credit Card", "Gold Credit Card", "Platinum Credit Card",
    "Titanium Credit Card", "World Elite Card", "Cashback Card",
    "Travel Rewards Card", "Business Credit Card",
]

STATUS_NAME_MAP = {
    "NORM": "Normal",
    "CLSB": "Closed by Bank",
    "CLSC": "Closed by Customer",
    "WROF": "Write Off",
    "90DA": "90 Days Delinquent",
    "30DD": "30 Days Delinquent",
    "SUSP": "Profit Suspended",
    "LOST": "Lost",
    "60DA": "60 Days Delinquent",
    "NEW": "New",
    "CNCD": "Cancelled",
    "FRAD": "Fraud",
    "CLSD": "Closed",
}

DELINQUENCY_MAP = {
    "NORM": 0, "CLSB": 0, "CLSC": 0, "WROF": 180, "90DA": 90,
    "30DD": 30, "SUSP": 120, "LOST": 0, "60DA": 60, "NEW": 0,
    "CNCD": 0, "FRAD": 0, "CLSD": 0,
}

ORGANIZATIONS = [
    "Vodafone Egypt", "Orange Egypt", "National Bank of Egypt",
    "CIB", "Banque Misr", "Orascom", "El Sewedy Electric",
    "Elsewedy Capital", "Talaat Moustafa Group", "Juhayna",
    "Edita Food Industries", "IBM Egypt", "Microsoft Egypt",
    "Amazon Egypt", "Freelancer", "Self Employed", "Government",
    "Ministry of Health", "Cairo University", "Military",
]

GENDERS = ["M", "F"]
CUSTOMER_TYPES = ["Primary", "Secondary"]

MCC_CODES = [
    5411, 5812, 5541, 5912, 5999, 7011, 4111, 5311, 5651, 5732,
    5814, 5691, 5944, 5945, 7230, 7298, 8011, 8021, 8099, 4900,
]

MERCH_NAMES = [
    "Carrefour Egypt", "Seoudi Market", "Spinneys", "B.TECH",
    "2B", "Jumia Egypt", "Amazon.eg", "Noon Egypt",
    "McDonald's", "KFC Egypt", "Costa Coffee", "Starbucks Egypt",
    "Uber Egypt", "Shell Egypt", "Total Energies", "Vodafone Cash",
    "Talabat", "Breadfast", "ElMenus", "Instashop",
    "H&M Egypt", "Zara Egypt", "Virgin Megastore", "iStyle",
    "LG Electronics", "Samsung Egypt", "Apple Store Online",
    "Booking.com", "Expedia", "Emirates Airlines",
]

SOURCES = ["POS", "ECOM", "ATM", "MOTO"]

STATUS_WEIGHTS = {
    "FEB": {"NORM": 60, "NEW": 10, "30DD": 5, "60DA": 3, "90DA": 2,
            "SUSP": 2, "WROF": 1, "CLSB": 4, "CLSC": 4, "CLSD": 2,
            "LOST": 2, "CNCD": 2, "FRAD": 1},
    "MAR": {"NORM": 55, "NEW": 8, "30DD": 7, "60DA": 4, "90DA": 3,
            "SUSP": 3, "WROF": 2, "CLSB": 5, "CLSC": 4, "CLSD": 2,
            "LOST": 2, "CNCD": 3, "FRAD": 1},
    "APR": {"NORM": 50, "NEW": 6, "30DD": 8, "60DA": 5, "90DA": 4,
            "SUSP": 4, "WROF": 3, "CLSB": 5, "CLSC": 5, "CLSD": 3,
            "LOST": 3, "CNCD": 3, "FRAD": 2},
    "MAY": {"NORM": 52, "NEW": 12, "30DD": 6, "60DA": 4, "90DA": 3,
            "SUSP": 3, "WROF": 2, "CLSB": 4, "CLSC": 5, "CLSD": 2,
            "LOST": 2, "CNCD": 3, "FRAD": 1},
}


def _weighted_status(month_key: str) -> str:
    weights = STATUS_WEIGHTS[month_key]
    statuses = list(weights.keys())
    return random.choices(statuses, weights=[weights[s] for s in statuses], k=1)[0]


def _pick_products(organization: str, credit_limit: int) -> list[str]:
    """Create correlated holdings so recommendation has real product signal."""
    n_products = random.choices([1, 2, 3, 4], weights=[58, 25, 12, 5], k=1)[0]
    held_products = []

    if organization == "Self Employed":
        held_products.append("Business Credit Card")
    elif credit_limit >= 100000:
        held_products.append(random.choice(["World Elite Card", "Platinum Credit Card"]))
    elif credit_limit >= 50000:
        held_products.append(random.choice(["Platinum Credit Card", "Titanium Credit Card"]))
    else:
        held_products.append(random.choice(["Classic Credit Card", "Gold Credit Card", "Cashback Card"]))

    if any(p in held_products for p in ["World Elite Card", "Platinum Credit Card"]):
        if random.random() < 0.8:
            held_products.append("Travel Rewards Card")
        if random.random() < 0.35:
            held_products.append("Titanium Credit Card")

    if any(p in held_products for p in ["Classic Credit Card", "Gold Credit Card"]):
        if random.random() < 0.7:
            held_products.append("Cashback Card")

    if "Business Credit Card" in held_products and random.random() < 0.55:
        held_products.append(random.choice(["Gold Credit Card", "Platinum Credit Card"]))

    held_products = list(dict.fromkeys(held_products))
    while len(held_products) < n_products:
        candidates = [p for p in PRODUCT_NAMES if p not in held_products]
        if not candidates:
            break
        held_products.append(random.choice(candidates))

    return held_products[:n_products]


def _build_customer_pool(rim_start: int, count: int) -> dict:
    pool = {}
    for i in range(count):
        rim = rim_start + i
        branch = random.choice(BRANCHES)
        dob = datetime(1960, 1, 1) + timedelta(days=random.randint(0, 365 * 40))
        creation = datetime(2018, 1, 1) + timedelta(days=random.randint(0, 365 * 7))
        credit_limit = random.choice([
            5000, 10000, 15000, 20000, 30000,
            50000, 75000, 100000, 150000, 200000,
        ])
        organization = random.choice(ORGANIZATIONS)

        pool[rim] = {
            "BRANCH_ID": branch[0],
            "BRANCH_NAME": branch[1],
            "CREATION_DATE": creation,
            "CREDIT_LIMIT": credit_limit,
            "NAMES": _pick_products(organization, credit_limit),
            "JOINING_FEE": random.choice([0, 100, 200, 500]),
            "ANNUAL_FEE": random.choice([0, 150, 300, 500, 1000]),
            "GENDER": random.choice(GENDERS),
            "DOB": dob,
            "ORGANIZATION": organization,
            "CUSTOMER_TYPE": random.choices(CUSTOMER_TYPES, weights=[80, 20], k=1)[0],
            "FIRST_REPLACED_CARD": random.choice(["", "", "", "VISA-OLD", "MC-OLD"]),
            "SECOND_REPLACED_CARD": random.choice(["", "", "", "", "VISA-OLD2"]),
            "THIRD_REPLACED_CARD": "",
        }
    return pool


def _generate_prime_month(
    rim_list: list[int],
    customer_pool: dict,
    month_key: str,
    snapshot_date: datetime,
) -> pd.DataFrame:
    rows = []
    for rim in rim_list:
        c = customer_pool[rim]
        for product_name in c["NAMES"]:
            status = _weighted_status(month_key)
            activated = "A" if status not in ("CLSB", "CLSC", "CLSD", "CNCD") else "I"
            credit_limit = c["CREDIT_LIMIT"]

            ledger = round(random.uniform(-credit_limit * 0.1, credit_limit * 0.95), 2)
            available = round(credit_limit - abs(ledger) - random.uniform(0, 500), 2)
            overdue = 0.0
            if status in ("30DD", "60DA", "90DA", "SUSP", "WROF"):
                overdue = round(random.uniform(500, credit_limit * 0.6), 2)

            last_pay_date = snapshot_date - timedelta(days=random.randint(1, 30))
            last_stmt_date = snapshot_date - timedelta(days=random.randint(0, 15))
            no_cycles = max(1, (snapshot_date - c["CREATION_DATE"]).days // 30)
            closure_date = None
            if status in ("CLSB", "CLSC", "CLSD", "WROF"):
                closure_date = snapshot_date - timedelta(days=random.randint(0, 60))

            rows.append({
                "BRANCH_ID": c["BRANCH_ID"],
                "BRANCH_NAME": c["BRANCH_NAME"],
                "CREATION_DATE": c["CREATION_DATE"].strftime("%Y-%m-%d"),
                "CREDIT_LIMIT": credit_limit,
                "ACTIVATED": activated,
                "STATUS": status,
                "STATUS_NAME": STATUS_NAME_MAP[status],
                "DELINQUENCY": DELINQUENCY_MAP[status],
                "LAST_STATEMENT_DATE": last_stmt_date.strftime("%Y-%m-%d"),
                "NAME": product_name,
                "JOINING_FEE": c["JOINING_FEE"],
                "ANNUAL_FEE": c["ANNUAL_FEE"],
                "LEDGER_BALANCE": ledger,
                "AVAILABLE_LIMIT": max(available, 0),
                "LAST_PAYMENT_AMOUNT": round(random.uniform(0, credit_limit * 0.3), 2),
                "LAST_PAYMENT_DATE": last_pay_date.strftime("%Y-%m-%d"),
                "TOTAL_HOLD": round(random.uniform(0, 2000), 2),
                "GENDER": c["GENDER"],
                "DOB": c["DOB"].strftime("%Y-%m-%d"),
                "ORGANIZATION": c["ORGANIZATION"],
                "CUSTOMER_TYPE": c["CUSTOMER_TYPE"],
                "RIM_NO": rim,
                "CLOSURE_DATE": closure_date.strftime("%Y-%m-%d") if closure_date else "",
                "OVERDUEAMOUNT": overdue,
                "NO_OF_CYCLES": no_cycles,
                "FIRST_REPLACED_CARD": c["FIRST_REPLACED_CARD"],
                "SECOND_REPLACED_CARD": c["SECOND_REPLACED_CARD"],
                "THIRD_REPLACED_CARD": c["THIRD_REPLACED_CARD"],
                "Card account status ": status,
            })
    return pd.DataFrame(rows)


def _generate_transactions(
    rim_list: list[int],
    customer_pool: dict,
    month_start: datetime,
    month_end: datetime,
) -> pd.DataFrame:
    rows = []
    for rim in rim_list:
        c = customer_pool[rim]
        n_txn = random.randint(0, 25)
        for _ in range(n_txn):
            trxn_date = month_start + timedelta(
                seconds=random.randint(0, int((month_end - month_start).total_seconds()))
            )
            post_date = trxn_date + timedelta(days=random.randint(0, 3))

            is_foreign = random.random() < 0.15
            if is_foreign:
                ccy = random.choice([840, 826, 978])
                orig_amt = round(random.uniform(5, 2000), 2)
                rate = random.uniform(30, 55) if ccy == 840 else random.uniform(35, 70)
                embedded_fee = round(orig_amt * random.uniform(0.01, 0.03), 2)
                billing_amt = round(orig_amt * rate + embedded_fee, 2)
                settlement_amt = orig_amt
                settlement_ccy = 840
                country = random.choice(["USA", "GBR", "ARE", "SAU", "TUR", "FRA", "DEU"])
            else:
                ccy = 818
                orig_amt = round(random.uniform(10, 15000), 2)
                embedded_fee = 0.0
                billing_amt = orig_amt
                settlement_amt = orig_amt
                settlement_ccy = 818
                country = "EGY"

            product_name = random.choice(c["NAMES"])
            rows.append({
                "DESCRIPTION": product_name,
                "RIMNO": rim,
                "POST DATE": post_date.strftime("%Y-%m-%d"),
                "TRXN DATE": trxn_date.strftime("%Y-%m-%d"),
                "CCY": ccy,
                "ORIG AMOUNT": orig_amt,
                "EMBEDDED _FEE": embedded_fee,
                "BILLING AMT": billing_amt,
                "MCC": random.choice(MCC_CODES),
                "MERCHNAME": random.choice(MERCH_NAMES),
                "MERCH ID": f"M{random.randint(100000, 999999)}",
                "SOURCES": random.choices(SOURCES, weights=[50, 30, 10, 10], k=1)[0],
                "SETTLEMENT AMT": settlement_amt,
                "SETTLEMENT CCY": settlement_ccy,
                "BANKBRANCH": c["BRANCH_NAME"],
                "TRXN COUNTRY": country,
                "REVERSAL FLAG": random.choices(["Y", "N"], weights=[3, 97], k=1)[0],
            })
    return pd.DataFrame(rows)


def main():
    print("=== Generating fake data ===\n")

    customer_pool = _build_customer_pool(rim_start=100000, count=N_BASE_CUSTOMERS)
    base_rims = list(customer_pool.keys())
    month_rims = {}

    prime_months = [
        ("FEB", datetime(2026, 2, 1), "FEB2026"),
        ("MAR", datetime(2026, 3, 1), "MAR2026"),
        ("APR", datetime(2026, 4, 1), "APR2026"),
    ]

    for month_key, snap_date, filename in prime_months:
        df = _generate_prime_month(base_rims, customer_pool, month_key, snap_date)
        out_path = os.path.join(PRIME_OUT, f"{filename}.csv")
        df.to_csv(out_path, index=False)
        month_rims[month_key] = list(base_rims)
        print(
            f"  [OK] Prime: {filename}.csv  ({len(df)} rows, "
            f"{df['RIM_NO'].nunique()} customers, "
            f"statuses: {df['STATUS'].value_counts().to_dict()})"
        )

    may_rims = list(base_rims)
    dropped = random.sample(may_rims, N_DROP_MAY)
    for rim in dropped:
        may_rims.remove(rim)

    new_pool = _build_customer_pool(rim_start=100000 + N_BASE_CUSTOMERS, count=N_NEW_MAY)
    customer_pool.update(new_pool)
    may_rims.extend(new_pool.keys())

    df_may = _generate_prime_month(may_rims, customer_pool, "MAY", datetime(2026, 5, 1))
    out_path = os.path.join(PRIME_OUT, "MAY2026.csv")
    df_may.to_csv(out_path, index=False)
    month_rims["MAY"] = may_rims
    print(
        f"  [OK] Prime: MAY2026.csv  ({len(df_may)} rows, "
        f"{df_may['RIM_NO'].nunique()} customers, dropped {N_DROP_MAY}, "
        f"added {N_NEW_MAY})"
    )
    print(f"    statuses: {df_may['STATUS'].value_counts().to_dict()}")

    txn_months = [
        ("FEB", datetime(2026, 2, 1), datetime(2026, 2, 28), "202602"),
        ("MAR", datetime(2026, 3, 1), datetime(2026, 3, 31), "202603"),
        ("APR", datetime(2026, 4, 1), datetime(2026, 4, 30), "202604"),
        ("MAY", datetime(2026, 5, 1), datetime(2026, 5, 31), "202605"),
    ]

    for month_key, month_start, month_end, filename in txn_months:
        df_txn = _generate_transactions(month_rims[month_key], customer_pool, month_start, month_end)
        out_path = os.path.join(TXN_OUT, f"{filename}.xlsx")
        df_txn.to_excel(out_path, index=False)
        print(
            f"  [OK] Txn:   {filename}.xlsx  ({len(df_txn)} rows, "
            f"{df_txn['RIMNO'].nunique()} unique customers)"
        )

    print("\n=== Done! ===")
    print(f"Prime  -> {PRIME_OUT}")
    print(f"Txn    -> {TXN_OUT}")


if __name__ == "__main__":
    main()
