#!/usr/bin/env python3
"""Fit per-segment frequency/severity from SQL and run Monte Carlo simulations.

This module:
1. Uses SQL GROUP BY + JOIN to fit a Poisson lambda and log-normal severity per segment
2. Runs vectorised Monte Carlo simulations, summing losses across segments
3. Stores each run's results in simulation_runs / simulation_results tables in the DB
4. Exports simulated_totals.csv for further analysis

Segments (home / auto / commercial) are defined in create_db.py.
"""
import argparse
import datetime
import sqlite3
import numpy as np
import pandas as pd


def load_segment_params(db_path: str) -> dict:
    """Return per-segment {lambda_per_policy, n_policies, mu, sigma} using SQL GROUP BY."""
    with sqlite3.connect(db_path) as conn:
        n_years = int(
            pd.read_sql_query("SELECT value FROM meta WHERE key='n_years'", conn).iloc[0, 0]
        )
        policy_counts = pd.read_sql_query(
            "SELECT segment, COUNT(*) as n_policies FROM policies GROUP BY segment", conn
        )
        claim_counts = pd.read_sql_query("""
            SELECT p.segment, COUNT(c.id) as n_claims
            FROM policies p
            LEFT JOIN claims c ON c.policy_id = p.policy_id
            GROUP BY p.segment
        """, conn)
        amounts_df = pd.read_sql_query("""
            SELECT p.segment, c.amount
            FROM claims c
            JOIN policies p ON c.policy_id = p.policy_id
        """, conn)

    merged = policy_counts.merge(claim_counts, on="segment")
    segment_params = {}

    for _, row in merged.iterrows():
        seg = row["segment"]
        n_policies = int(row["n_policies"])
        lambda_pp = int(row["n_claims"]) / (n_policies * n_years)

        seg_amounts = amounts_df[amounts_df["segment"] == seg]["amount"].values.astype(float)
        if len(seg_amounts) == 0:
            mu, sigma = 0.0, 1.0
        else:
            mu = float(np.mean(np.log(seg_amounts)))
            sigma = float(np.std(np.log(seg_amounts), ddof=1))

        segment_params[seg] = dict(lambda_per_policy=lambda_pp, n_policies=n_policies, mu=mu, sigma=sigma)

    return segment_params


def run_simulation(segment_params: dict, portfolio_size: int | None, n_sims: int, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    total_historical = sum(p["n_policies"] for p in segment_params.values())
    sim_totals = np.zeros(n_sims, dtype=float)

    for seg, params in segment_params.items():
        # Scale each segment proportionally to its historical share of the portfolio
        share = params["n_policies"] / total_historical
        seg_size = round(portfolio_size * share) if portfolio_size else params["n_policies"]
        lam = params["lambda_per_policy"] * seg_size

        n_claims = rng.poisson(lam, size=n_sims)
        mask = n_claims > 0
        if mask.any():
            all_samples = rng.lognormal(mean=params["mu"], sigma=params["sigma"], size=int(n_claims[mask].sum()))
            splits = np.split(all_samples, np.cumsum(n_claims[mask])[:-1])
            seg_totals = np.zeros(n_sims)
            seg_totals[mask] = [s.sum() for s in splits]
            sim_totals += seg_totals

    return sim_totals


def summarize_sim(sim_totals: np.ndarray, quantiles=(0.5, 0.9, 0.95, 0.99)) -> dict:
    qs = {f"var_{int(q*100)}": float(np.quantile(sim_totals, q)) for q in quantiles}
    return {
        "mean_loss": float(sim_totals.mean()),
        "volatility": float(sim_totals.std(ddof=1)),
        **qs,
        "min_loss": float(np.min(sim_totals)),
        "p25_loss": float(np.percentile(sim_totals, 25)),
        "p75_loss": float(np.percentile(sim_totals, 75)),
        "max_loss": float(np.max(sim_totals)),
    }


def save_simulation(db_path: str, seed: int | None, n_sims: int, portfolio_size: int, sim_totals: np.ndarray, summary: dict) -> int:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS simulation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                seed INTEGER,
                n_sims INTEGER NOT NULL,
                portfolio_size INTEGER NOT NULL,
                mean_loss REAL,
                volatility REAL,
                var_90 REAL,
                var_95 REAL,
                var_99 REAL
            );
            CREATE TABLE IF NOT EXISTS simulation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES simulation_runs(id),
                total_loss REAL NOT NULL
            );
        """)
        cur.execute("""
            INSERT INTO simulation_runs
                (created_at, seed, n_sims, portfolio_size, mean_loss, volatility, var_90, var_95, var_99)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            seed, n_sims, portfolio_size,
            summary["mean_loss"], summary["volatility"],
            summary["var_90"], summary["var_95"], summary["var_99"],
        ))
        run_id = cur.lastrowid
        cur.executemany(
            "INSERT INTO simulation_results (run_id, total_loss) VALUES (?, ?)",
            [(run_id, float(v)) for v in sim_totals],
        )
        conn.commit()
    return run_id


def main():
    p = argparse.ArgumentParser(description="Monte Carlo simulation of portfolio claims losses")
    p.add_argument("--db", default="claims.db")
    p.add_argument("--n-sims", type=int, default=10000)
    p.add_argument("--portfolio-size", type=int, default=None, help="Total policies to simulate (defaults to historical size)")
    p.add_argument("--out-csv", default="simulated_totals.csv")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    segment_params = load_segment_params(args.db)

    print("Fitted segment parameters:")
    for seg, params in segment_params.items():
        print(f"  {seg:12s} lambda={params['lambda_per_policy']:.4f}  mu={params['mu']:.3f}  sigma={params['sigma']:.3f}  n_policies={params['n_policies']}")

    total_policies = sum(p["n_policies"] for p in segment_params.values())
    portfolio_size = args.portfolio_size or total_policies

    rng = np.random.default_rng(args.seed)
    sim_totals = run_simulation(segment_params, portfolio_size, args.n_sims, rng=rng)
    summary = summarize_sim(sim_totals)

    print("\nPortfolio Risk Metrics:")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k}: {v:,.2f}")

    run_id = save_simulation(args.db, args.seed, args.n_sims, portfolio_size, sim_totals, summary)
    print(f"\nSaved run #{run_id} to {args.db}")

    pd.Series(sim_totals, name="total_loss").to_csv(args.out_csv, index=False)
    print(f"Wrote {len(sim_totals)} simulation results to {args.out_csv}")


if __name__ == "__main__":
    main()
