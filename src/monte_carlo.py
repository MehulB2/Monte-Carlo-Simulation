#!/usr/bin/env python3
"""Fit frequency/severity from the SQL DB and run Monte Carlo simulations.

Exports a CSV `simulated_totals.csv` containing total portfolio loss per simulation.
"""
import argparse
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats


def load_claims(db_path: str):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT id, policy_id, claim_date, amount FROM claims", conn)
    # load meta
    meta = dict(pd.read_sql_query("SELECT key, value FROM meta", conn).values)
    conn.close()
    return df, meta


def fit_frequency(df: pd.DataFrame, meta: dict):
    # Compute lambda_per_policy empirically if meta not provided
    n_claims = len(df)
    if "n_policies" in meta and "n_years" in meta:
        n_policies = int(meta["n_policies"])
        n_years = int(meta["n_years"])
        lambda_per_policy = n_claims / (n_policies * n_years)
    elif "lambda_per_policy" in meta:
        lambda_per_policy = float(meta["lambda_per_policy"])
    else:
        raise RuntimeError("Meta data missing: cannot infer lambda_per_policy. Provide meta or precomputed value.")
    return lambda_per_policy


def fit_severity(df: pd.DataFrame):
    amounts = df["amount"].values.astype(float)
    if len(amounts) == 0:
        # fallback tiny distribution
        return None
    # Fit gamma with loc=0 (positive amounts)
    a, loc, scale = stats.gamma.fit(amounts, floc=0)
    return dict(dist_name="gamma", a=float(a), loc=float(loc), scale=float(scale))


def run_simulation(lambda_per_policy: float, severity_params: dict, portfolio_size: int, n_sims: int, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    sim_totals = np.zeros(n_sims, dtype=float)
    # effective lambda for the portfolio
    lam = lambda_per_policy * portfolio_size
    if severity_params is None:
        # assume zero or tiny
        for i in range(n_sims):
            n = rng.poisson(lam)
            sim_totals[i] = 0.0
        return sim_totals

    a = severity_params["a"]
    scale = severity_params["scale"]
    for i in range(n_sims):
        n = rng.poisson(lam)
        if n == 0:
            sim_totals[i] = 0.0
            continue
        # sample severities from fitted gamma
        samples = stats.gamma(a, scale=scale).rvs(size=n, random_state=rng)
        sim_totals[i] = samples.sum()
    return sim_totals


def summarize_sim(sim_totals: np.ndarray, quantiles=(0.5, 0.9, 0.95, 0.99)):
    mean = float(sim_totals.mean())
    std = float(sim_totals.std(ddof=1))
    qs = {f"q_{int(q*100)}": float(np.quantile(sim_totals, q)) for q in quantiles}
    # CVaR at q = expected shortfall: mean of losses >= VaR
    cvars = {}
    for q in quantiles:
        var = np.quantile(sim_totals, q)
        tail = sim_totals[sim_totals >= var]
        cvar = float(tail.mean()) if len(tail) > 0 else var
        cvars[f"cvar_{int(q*100)}"] = cvar

    out = {"mean": mean, "std": std}
    out.update(qs)
    out.update(cvars)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="claims.db", help="SQLite DB path")
    p.add_argument("--n-sims", type=int, default=10000)
    p.add_argument("--portfolio-size", type=int, default=None, help="Portfolio size to simulate (defaults to historical n_policies)")
    p.add_argument("--out-csv", default="simulated_totals.csv")
    args = p.parse_args()

    df, meta = load_claims(args.db)
    lambda_per_policy = fit_frequency(df, meta)
    sev = fit_severity(df)

    if args.portfolio_size is None and "n_policies" in meta:
        portfolio_size = int(meta["n_policies"])
    elif args.portfolio_size is not None:
        portfolio_size = args.portfolio_size
    else:
        raise RuntimeError("Provide --portfolio-size or include n_policies in DB meta")

    print(f"Estimated lambda_per_policy={lambda_per_policy:.6f}")
    if sev is None:
        print("No historical severities found; severity fit skipped.")
    else:
        print(f"Fitted severity: {sev}")

    sim_totals = run_simulation(lambda_per_policy, sev, portfolio_size, args.n_sims)
    summary = summarize_sim(sim_totals)
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:,.2f}")

    # Save raw simulation totals
    pd.Series(sim_totals, name="total_loss").to_csv(args.out_csv, index=False)
    print(f"Wrote simulation totals to {args.out_csv}")


if __name__ == "__main__":
    main()
