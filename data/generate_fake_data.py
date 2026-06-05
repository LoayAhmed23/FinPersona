"""
Generate fake Prime and Transaction data for local testing.

Outputs
-------
data/original/prime/FEB2026.xlsx, MAR2026.xlsx, APR2026.xlsx, MAY2026.xlsx
data/original/transaction/202503.xlsx, 202504.xlsx, 202505.xlsx

Run:  python data/generate_fake_data.py
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─── reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRIME_OUT = os.path.join(SCRIPT_DIR, "original", "prime")
TXN_OUT = os.path.join(SCRIPT_DIR, "original", "transaction")
os.makedirs(PRIME_OUT, exist_ok=True)
os.makedirs(TXN_OUT, exist_ok=True)

# ─── constants ──────────────────────────────────────────────────────────────
N_BASE_CUSTOMERS = 1000
N_NEW_MAY = 80          # new customers joining in MAY
N_DROP_MAY = 5          # exactly 5 customers dropping off for churn

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

STATUSES_ALL = [
    "NORM", "CLSB", "CLSC", "WROF", "90DA", "30DD", "SUSP",
    "LOST", "60DA", "NEW", "CNCD", "FRAD", "CLSD",
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
    "NEW":  "New",
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
ACTIVATED_FLAGS = ["A", "I"]

# ─── merchant / transaction constants ───────────────────────────────────────
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
COUNTRIES = ["EGY", "USA", "GBR", "ARE", "SAU", "TUR", "FRA", "DEU"]
DESCRIPTIONS = [
    "Classic Credit Card", "Gold Credit Card", "Platinum Credit Card",
    "Titanium Credit Card", "World Elite Card",
]


# ═══════════════════════════════════════════════════════════════════════════
#  STATUS DISTRIBUTION PER MONTH  (to simulate variation)
# ═══════════════════════════════════════════════════════════════════════════
# Weights for each status in each month — they shift over time
STATUS_WEIGHTS = {
    "FEB": {"NORM": 90, "NEW": 5, "30DD": 3, "60DA": 1, "90DA": 1,
            "SUSP": 0, "WROF": 0, "CLSB": 0, "CLSC": 0, "CLSD": 0,
            "LOST": 0, "CNCD": 0, "FRAD": 0},
    "MAR": {"NORM": 90, "NEW": 5, "30DD": 3, "60DA": 1, "90DA": 1,
            "SUSP": 0, "WROF": 0, "CLSB": 0, "CLSC": 0, "CLSD": 0,
            "LOST": 0, "CNCD": 0, "FRAD": 0},
    "APR": {"NORM": 90, "NEW": 5, "30DD": 3, "60DA": 1, "90DA": 1,
            "SUSP": 0, "WROF": 0, "CLSB": 0, "CLSC": 0, "CLSD": 0,
            "LOST": 0, "CNCD": 0, "FRAD": 0},
    "MAY": {"NORM": 90, "NEW": 5, "30DD": 3, "60DA": 1, "90DA": 1,
            "SUSP": 0, "WROF": 0, "CLSB": 0, "CLSC": 0, "CLSD": 0,
            "LOST": 0, "CNCD": 0, "FRAD": 0},
}


def _weighted_status(month_key: str) -> str:
    """Pick a random status according to the month's weight distribution."""
    w = STATUS_WEIGHTS[month_key]
    statuses = list(w.keys())
    weights = [w[s] for s in statuses]
    return random.choices(statuses, weights=weights, k=1)[0]


# ═══════════════════════════════════════════════════════════════════════════
#  CUSTOMER SEED DATA  (static attributes that stay the same across months)
# ═══════════════════════════════════════════════════════════════════════════
def _build_customer_pool(rim_start: int, count: int) -> dict:
    """Create a dict of RIM_NO -> static attributes."""
    pool = {}
    for i in range(count):
        rim = rim_start + i
        branch = random.choice(BRANCHES)
        dob = datetime(1960, 1, 1) + timedelta(days=random.randint(0, 365 * 40))
        creation = datetime(2018, 1, 1) + timedelta(days=random.randint(0, 365 * 7))
        credit_limit = random.choice([5000, 10000, 15000, 20000, 30000,
                                      50000, 75000, 100000, 150000, 200000])
        organization = random.choice(ORGANIZATIONS)

        n_products = random.choices([1, 2, 3], weights=[70, 20, 10])[0]
        
        # Correlate initial product based on demographics
        held_products = []
        if organization == "Self Employed":
            held_products.append("Business Credit Card")
        elif credit_limit >= 100000:
            held_products.append(random.choice(["World Elite Card", "Platinum Credit Card"]))
        elif credit_limit >= 50000:
            held_products.append(random.choice(["Platinum Credit Card", "Titanium Credit Card"]))
        else:
            held_products.append(random.choice(["Classic Credit Card", "Gold Credit Card", "Cashback Card"]))
            
        # Product-to-Product correlation for CBF
        if n_products > 1:
            if "World Elite Card" in held_products or "Platinum Credit Card" in held_products:
                if random.random() < 0.8:  # 80% chance to co-occur
                    held_products.append("Travel Rewards Card")
            if "Classic Credit Card" in held_products or "Gold Credit Card" in held_products:
                if random.random() < 0.7:  # 70% chance to co-occur
                    held_products.append("Cashback Card")
        
        # Fill remaining if needed
        while len(held_products) < n_products:
            candidates = [p for p in PRODUCT_NAMES if p not in held_products]
            if not candidates:
                break
            held_products.append(random.choice(candidates))
            
        held_products = list(set(held_products))[:n_products]
        
        pool[rim] = {
            "BRANCH_ID": branch[0],
            "BRANCH_NAME": branch[1],
            "CREATION_DATE": creation,
            "CREDIT_LIMIT": credit_limit,
            "NAMES": held_products,
            "JOINING_FEE": random.choice([0, 100, 200, 500]),
            "ANNUAL_FEE": random.choice([0, 150, 300, 500, 1000]),
            "GENDER": random.choice(GENDERS),
            "DOB": dob,
            "ORGANIZATION": organization,
            "CUSTOMER_TYPE": random.choices(CUSTOMER_TYPES, weights=[80, 20])[0],
            "FIRST_REPLACED_CARD": random.choice(["", "", "", "VISA-OLD", "MC-OLD"]),
            "SECOND_REPLACED_CARD": random.choice(["", "", "", "", "VISA-OLD2"]),
            "THIRD_REPLACED_CARD": "",
        }
    return pool


def _generate_prime_month(rim_list: list, customer_pool: dict,
                          month_key: str, snapshot_date: datetime) -> pd.DataFrame:
    """Build a single month's prime DataFrame."""
    rows = []
    for rim in rim_list:
        c = customer_pool[rim]
        for prod_name in c["NAMES"]:
            status = _weighted_status(month_key)
            activated = "A" if status not in ("CLSB", "CLSC", "CLSD", "CNCD") else "I"
            credit_limit = c["CREDIT_LIMIT"]
            # financial fields that change monthly
            ledger = round(random.uniform(-credit_limit * 0.1, credit_limit * 0.95), 2)
            available = round(credit_limit - abs(ledger) - random.uniform(0, 500), 2)
            available = max(available, 0)
            overdue = 0.0
            if status in ("30DD", "60DA", "90DA", "SUSP", "WROF"):
                overdue = round(random.uniform(500, credit_limit * 0.6), 2)
            last_pay_amt = round(random.uniform(0, credit_limit * 0.3), 2)
            last_pay_date = snapshot_date - timedelta(days=random.randint(1, 30))
            last_stmt_date = snapshot_date - timedelta(days=random.randint(0, 15))
            total_hold = round(random.uniform(0, 2000), 2)
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
                "STATUS": status.upper(),
                "STATUS_NAME": STATUS_NAME_MAP.get(status.upper(), status),
                "DELINQUENCY": DELINQUENCY_MAP.get(status.upper(), 0),
                "LAST_STATEMENT_DATE": last_stmt_date.strftime("%Y-%m-%d"),
                "NAME": prod_name,
                "JOINING_FEE": c["JOINING_FEE"],
                "ANNUAL_FEE": c["ANNUAL_FEE"],
                "LEDGER_BALANCE": ledger,
                "AVAILABLE_LIMIT": available,
                "LAST_PAYMENT_AMOUNT": last_pay_amt,
                "LAST_PAYMENT_DATE": last_pay_date.strftime("%Y-%m-%d"),
                "TOTAL_HOLD": total_hold,
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
                "Card account status ": status.upper(),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
#  TRANSACTION GENERATION
# ═══════════════════════════════════════════════════════════════════════════
def _generate_transactions(rim_list: list, customer_pool: dict,
                           month_start: datetime, month_end: datetime) -> pd.DataFrame:
    """Generate random transactions for the given RIMNOs within a date range."""
    rows = []
    for rim in rim_list:
        c = customer_pool[rim]
        n_txn = random.randint(0, 25)  # 0-25 transactions per customer per month
        for _ in range(n_txn):
            trxn_date = month_start + timedelta(
                seconds=random.randint(0, int((month_end - month_start).total_seconds()))
            )
            post_date = trxn_date + timedelta(days=random.randint(0, 3))

            is_foreign = random.random() < 0.15
            if is_foreign:
                ccy = random.choice([840, 826, 978])  # USD, GBP, EUR
                orig_amt = round(random.uniform(5, 2000), 2)
                rate = random.uniform(30, 55) if ccy == 840 else random.uniform(35, 70)
                embedded_fee = round(orig_amt * random.uniform(0.01, 0.03), 2)
                billing_amt = round(orig_amt * rate + embedded_fee, 2)
                settlement_amt = round(orig_amt, 2)  # in USD
                settlement_ccy = 840
                country = random.choice(["USA", "GBR", "ARE", "SAU", "TUR", "FRA", "DEU"])
            else:
                ccy = 818  # EGP
                orig_amt = round(random.uniform(10, 15000), 2)
                embedded_fee = 0.0
                billing_amt = orig_amt
                settlement_amt = orig_amt
                settlement_ccy = 818
                country = "EGY"

            mcc = random.choice(MCC_CODES)
            merch = random.choice(MERCH_NAMES)
            merch_id = f"M{random.randint(100000, 999999)}"
            source = random.choices(SOURCES, weights=[50, 30, 10, 10])[0]
            reversal = random.choices(["Y", "N"], weights=[3, 97])[0]
            prod_name = random.choice(c["NAMES"])

            rows.append({
                "DESCRIPTION": prod_name,
                "RIMNO": rim,
                "POST DATE": post_date.strftime("%Y-%m-%d"),
                "TRXN DATE": trxn_date.strftime("%Y-%m-%d"),
                "CCY": ccy,
                "ORIG AMOUNT": orig_amt,
                "EMBEDDED _FEE": embedded_fee,
                "BILLING AMT": billing_amt,
                "MCC": mcc,
                "MERCHNAME": merch,
                "MERCH ID": merch_id,
                "SOURCES": source,
                "SETTLEMENT AMT": settlement_amt,
                "SETTLEMENT CCY": settlement_ccy,
                "BANKBRANCH": c["BRANCH_NAME"],
                "TRXN COUNTRY": country,
                "REVERSAL FLAG": reversal,
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=== Generating fake data ===\n")

    # --- 1. Build base customer pool ---
    customer_pool = _build_customer_pool(rim_start=100000, count=N_BASE_CUSTOMERS)
    base_rims = list(customer_pool.keys())

    # --- 2. Prime months FEB–APR 2026 (stable ~1000 customers) ---
    prime_months = [
        ("FEB", datetime(2026, 2, 1), "FEB2026"),
        ("MAR", datetime(2026, 3, 1), "MAR2026"),
        ("APR", datetime(2026, 4, 1), "APR2026"),
    ]

    month_rims = {}  # track which RIMs appear in each month for transactions

    for month_key, snap_date, filename in prime_months:
        df = _generate_prime_month(base_rims, customer_pool, month_key, snap_date)
        out_path = os.path.join(PRIME_OUT, f"{filename}.csv")
        df.to_csv(out_path, index=False)
        month_rims[month_key] = list(base_rims)
        print(f"  [OK] Prime: {filename}.csv  ({len(df)} rows, "
              f"statuses: {df['STATUS'].value_counts().to_dict()})")

    # --- 3. MAY 2026: drop some customers, add new ones ---
    may_rims = list(base_rims)
    dropped = random.sample(may_rims, N_DROP_MAY)
    for r in dropped:
        may_rims.remove(r)

    new_pool = _build_customer_pool(
        rim_start=100000 + N_BASE_CUSTOMERS, count=N_NEW_MAY
    )
    customer_pool.update(new_pool)
    may_rims.extend(new_pool.keys())

    df_may = _generate_prime_month(may_rims, customer_pool, "MAY", datetime(2026, 5, 1))
    out_path = os.path.join(PRIME_OUT, "MAY2026.csv")
    df_may.to_csv(out_path, index=False)
    month_rims["MAY"] = may_rims
    print(f"  [OK] Prime: MAY2026.csv  ({len(df_may)} rows, "
          f"dropped {N_DROP_MAY}, added {N_NEW_MAY} new customers)")
    print(f"    statuses: {df_may['STATUS'].value_counts().to_dict()}")

    # --- 4. Transaction files ---
    txn_months = [
        ("FEB", datetime(2026, 2, 1), datetime(2026, 2, 28), "202602"),
        ("MAR", datetime(2026, 3, 1), datetime(2026, 3, 31), "202603"),
        ("APR", datetime(2026, 4, 1), datetime(2026, 4, 30), "202604"),
        ("MAY", datetime(2026, 5, 1), datetime(2026, 5, 31), "202605"),
    ]

    for month_key, m_start, m_end, filename in txn_months:
        rims = month_rims[month_key]
        df_txn = _generate_transactions(rims, customer_pool, m_start, m_end)
        out_path = os.path.join(TXN_OUT, f"{filename}.xlsx")
        df_txn.to_excel(out_path, index=False)
        print(f"  [OK] Txn:   {filename}.xlsx  ({len(df_txn)} rows, "
              f"{df_txn['RIMNO'].nunique()} unique customers)")

    print("\n=== Done! ===")
    print(f"Prime  -> {PRIME_OUT}")
    print(f"Txn    -> {TXN_OUT}")


if __name__ == "__main__":
    main()
