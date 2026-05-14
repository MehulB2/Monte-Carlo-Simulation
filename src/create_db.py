#!/usr/bin/env python3
"""Create a sample SQLite DB with synthetic segmented claims history.

Schema:
  policies(policy_id, segment)
  claims(id, policy_id, claim_date, amount)
  meta(key, value)
"""
import argparse
import sqlite3
import datetime
import numpy as np

SEGMENTS = [
    dict(name="home",       lambda_mult=1.0, sev_mu=8.5,  sev_sigma=1.0),
    dict(name="auto",       lambda_mult=1.5, sev_mu=7.5,  sev_sigma=0.8),
    dict(name="commercial", lambda_mult=0.5, sev_mu=10.0, sev_sigma=1.2),
]


def create_db(path: str, n_policies: int, n_years: int, base_lambda: float, seed: int = 42):
    rng = np.random.default_rng(seed)

    with sqlite3.connect(path) as conn:
        cur = conn.cursor()

        cur.executescript("""
            DROP TABLE IF EXISTS claims;
            DROP TABLE IF EXISTS policies;
            DROP TABLE IF EXISTS meta;
            CREATE TABLE policies (
                policy_id INTEGER PRIMARY KEY,
                segment TEXT NOT NULL
            );
            CREATE TABLE claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                policy_id INTEGER NOT NULL REFERENCES policies(policy_id),
                claim_date TEXT NOT NULL,
                amount REAL NOT NULL
            );
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)

        n_segs = len(SEGMENTS)
        base_per_seg = n_policies // n_segs
        start_year = datetime.date.today().year - n_years
        policy_id = 1
        total_claims = 0

        for i, seg in enumerate(SEGMENTS):
            seg_n = base_per_seg + (1 if i < n_policies % n_segs else 0)
            lam = base_lambda * seg["lambda_mult"]

            for _ in range(seg_n):
                cur.execute("INSERT INTO policies VALUES (?, ?)", (policy_id, seg["name"]))

                for year_offset in range(n_years):
                    year = start_year + year_offset
                    for _ in range(rng.poisson(lam)):
                        amount = float(rng.lognormal(mean=seg["sev_mu"], sigma=seg["sev_sigma"]))
                        day = int(rng.integers(1, 366))
                        claim_date = datetime.date(year, 1, 1) + datetime.timedelta(days=day - 1)
                        cur.execute(
                            "INSERT INTO claims (policy_id, claim_date, amount) VALUES (?, ?, ?)",
                            (policy_id, claim_date.isoformat(), amount),
                        )
                        total_claims += 1

                policy_id += 1

        cur.executemany("REPLACE INTO meta VALUES (?, ?)", [
            ("n_policies", str(n_policies)),
            ("n_years",    str(n_years)),
            ("base_lambda", str(base_lambda)),
        ])
        conn.commit()

    print(f"Wrote '{path}': {n_policies} policies across {n_segs} segments, {total_claims} claims.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="claims.db")
    p.add_argument("--n-policies", type=int, default=1000)
    p.add_argument("--n-years", type=int, default=5)
    p.add_argument("--lambda-per-policy", type=float, default=0.1, help="Base claim rate (multiplied per segment)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    create_db(args.out, args.n_policies, args.n_years, args.lambda_per_policy, seed=args.seed)


if __name__ == "__main__":
    main()
