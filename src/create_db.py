#!/usr/bin/env python3
"""Create a sample SQLite DB with synthetic claims history.

Produces a `claims` table with columns: id, policy_id, claim_date, amount.
Also writes metadata into a `meta` table for `n_policies` and `n_years`.
"""
import argparse
import sqlite3
import datetime
import numpy as np


def create_db(path: str, n_policies: int, n_years: int, lambda_per_policy: float, sev_mu: float, sev_sigma: float, seed: int = 42):
    rng = np.random.default_rng(seed)  # reproducible random generation
    conn = sqlite3.connect(path)  # open/create database
    cur = conn.cursor()  # database cursor
    cur.execute("""
    CREATE TABLE IF NOT EXISTS claims (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        policy_id INTEGER NOT NULL,
        claim_date TEXT NOT NULL,
        amount REAL NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)

    total_inserted = 0
    start_year = datetime.date.today().year - n_years  # backdate n years
    for year_offset in range(n_years):  # iterate over historical years
        year = start_year + year_offset
        for policy_id in range(1, n_policies + 1):  # for each policy
            c = rng.poisson(lambda_per_policy)  # Poisson-sampled claim count
            for _ in range(c):  # for each claim this year
                amount = float(rng.lognormal(mean=sev_mu, sigma=sev_sigma))  # sample severity
                day_of_year = rng.integers(1, 366)  # random day in year
                claim_date = datetime.date(year, 1, 1) + datetime.timedelta(days=int(day_of_year - 1))  # generate claim date
                cur.execute("INSERT INTO claims (policy_id, claim_date, amount) VALUES (?, ?, ?)", (policy_id, claim_date.isoformat(), amount))
                total_inserted += 1

    cur.execute("REPLACE INTO meta (key, value) VALUES (?, ?)", ("n_policies", str(n_policies)))  # save portfolio size
    cur.execute("REPLACE INTO meta (key, value) VALUES (?, ?)", ("n_years", str(n_years)))  # save period
    cur.execute("REPLACE INTO meta (key, value) VALUES (?, ?)", ("lambda_per_policy", str(lambda_per_policy)))  # save frequency

    conn.commit()  # save changes
    conn.close()  # close connection
    print(f"Wrote DB '{path}' with {total_inserted} synthetic claims (n_policies={n_policies}, n_years={n_years}).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="claims.db", help="Output SQLite DB path")
    p.add_argument("--n-policies", type=int, default=1000)
    p.add_argument("--n-years", type=int, default=5)
    p.add_argument("--lambda-per-policy", type=float, default=0.1, help="Avg claims per policy-year")
    p.add_argument("--sev-mu", type=float, default=8.5, help="Log-normal mean (on log scale)")
    p.add_argument("--sev-sigma", type=float, default=1.0, help="Log-normal sigma (on log scale)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    create_db(args.out, args.n_policies, args.n_years, args.lambda_per_policy, args.sev_mu, args.sev_sigma, seed=args.seed)


if __name__ == "__main__":
    main()
