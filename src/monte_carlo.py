#!/usr/bin/env python3
"""Fit frequency/severity from the SQL DB and run Monte Carlo simulations.

This module:
1. Loads historical claim data from SQLite database
2. Fits a Poisson frequency distribution (claims per policy per year)
3. Fits a Gamma severity distribution (claim amount distribution)
4. Runs Monte Carlo simulations to generate portfolio loss distribution
5. Computes portfolio-level risk metrics: VaR, percentiles, mean, volatility

Exports a CSV `simulated_totals.csv` containing total portfolio loss per simulation.
"""
import argparse
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats


def load_claims(db_path: str):
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Query historical claim records
    df = pd.read_sql_query("SELECT id, policy_id, claim_date, amount FROM claims", conn)
    
    # Query metadata table (stores portfolio info and parameters)
    meta = dict(pd.read_sql_query("SELECT key, value FROM meta", conn).values)
    
    # Close connection
    conn.close()
    return df, meta


def fit_frequency(df: pd.DataFrame, meta: dict):
    # Count total historical claims
    n_claims = len(df)
    
    # Try to compute lambda from portfolio history
    if "n_policies" in meta and "n_years" in meta:
        # Empirical: lambda = total_claims / (policies * years)
        n_policies = int(meta["n_policies"])
        n_years = int(meta["n_years"])
        lambda_per_policy = n_claims / (n_policies * n_years)
    # Or use precomputed lambda from metadata
    elif "lambda_per_policy" in meta:
        lambda_per_policy = float(meta["lambda_per_policy"])
    else:
        raise RuntimeError("Meta data missing: cannot infer lambda_per_policy. Provide meta or precomputed value.")
    
    return lambda_per_policy


def fit_severity(df: pd.DataFrame):
    # Extract claim amounts and convert to float
    amounts = df["amount"].values.astype(float)
    
    # Handle empty dataset
    if len(amounts) == 0:
        return None
    
    # Fit Gamma distribution: scipy.stats.gamma.fit(data, floc=0)
    # floc=0 ensures location parameter is 0 (claims must be positive)
    # Returns: shape parameter (a), location (loc=0), scale parameter
    a, loc, scale = stats.gamma.fit(amounts, floc=0)
    
    # Return parameters as dictionary for later use in simulations
    return dict(dist_name="gamma", a=float(a), loc=float(loc), scale=float(scale))


def run_simulation(lambda_per_policy: float, severity_params: dict | None, portfolio_size: int, n_sims: int, rng=None):
    # Initialize random generator if not provided
    if rng is None:
        rng = np.random.default_rng()
    
    # Pre-allocate array to store total loss for each simulation
    sim_totals = np.zeros(n_sims, dtype=float)
    
    # Compute effective portfolio-level Poisson parameter
    # (claims per policy per year) × (number of policies) = total expected claims
    lam = lambda_per_policy * portfolio_size
    
    # Handle case where no severity data available (no historical claims)
    if severity_params is None:
        # If no severity data, assume zero loss each simulation
        for i in range(n_sims):
            n = rng.poisson(lam)
            sim_totals[i] = 0.0
        return sim_totals
    
    # Extract Gamma distribution parameters
    a = severity_params["a"]
    scale = severity_params["scale"]
    
    # Run Monte Carlo simulation for each scenario
    for i in range(n_sims):
        # Step 1: Sample number of claims this year from Poisson distribution
        n = rng.poisson(lam)
        
        # Step 2: If no claims, loss is 0
        if n == 0:
            sim_totals[i] = 0.0
            continue
        
        # Step 3: Sample individual claim amounts from fitted Gamma distribution
        # Repeat n times to get n claim amounts
        samples = stats.gamma(a, scale=scale).rvs(size=n, random_state=rng)
        
        # Step 4: Sum all claims to get total portfolio loss this year
        sim_totals[i] = samples.sum()
    
    return sim_totals


def summarize_sim(sim_totals: np.ndarray, quantiles=(0.5, 0.9, 0.95, 0.99)):

    # ========== CENTRAL TENDENCY ==========
    # Mean: Expected value of portfolio loss
    mean = float(sim_totals.mean())
    
    # Volatility: Standard deviation of losses (measures risk dispersion)
    # ddof=1 uses sample standard deviation (n-1 denominator)
    std = float(sim_totals.std(ddof=1))
    
    # ========== VALUE AT RISK (VaR) ==========
    # VaR at confidence level q: the loss amount L such that P(loss > L) = 1 - q
    # Example: VaR at 0.95 = 95th percentile = worst case with 95% confidence
    qs = {}
    for q in quantiles:
        # np.quantile computes percentile; q=0.95 returns 95th percentile value
        var_q = float(np.quantile(sim_totals, q))
        qs[f"var_{int(q*100)}"] = var_q
    
    # ========== LOSS PERCENTILES ==========
    # Provide additional context: min, 25th, 75th percentiles
    percentile_stats = {
        "min_loss": float(np.min(sim_totals)),
        "p25_loss": float(np.percentile(sim_totals, 25)),
        "p75_loss": float(np.percentile(sim_totals, 75)),
        "max_loss": float(np.max(sim_totals)),
    }
    
    # Assemble all metrics into output dictionary
    out = {"mean_loss": mean, "volatility": std}
    out.update(qs)  # Add VaR metrics
    out.update(percentile_stats)  # Add percentiles
    
    return out


def main():

    # ========== PARSE COMMAND-LINE ARGUMENTS ==========
    p = argparse.ArgumentParser(
        description="Monte Carlo simulation of portfolio claims losses"
    )
    p.add_argument("--db", default="claims.db", help="SQLite DB path")
    p.add_argument("--n-sims", type=int, default=10000, help="Number of simulations to run")
    p.add_argument(
        "--portfolio-size",
        type=int,
        default=None,
        help="Portfolio size to simulate (defaults to historical n_policies)"
    )
    p.add_argument("--out-csv", default="simulated_totals.csv", help="Output CSV path")
    args = p.parse_args()

    # ========== LOAD DATA ==========
    # Load historical claims and portfolio metadata from database
    df, meta = load_claims(args.db)
    
    # ========== FIT DISTRIBUTIONS ==========
    # Fit Poisson parameter: claims per policy per year
    lambda_per_policy = fit_frequency(df, meta)
    
    # Fit Gamma distribution to claim amounts (severity)
    sev = fit_severity(df)

    # ========== DETERMINE PORTFOLIO SIZE ==========
    # Use provided --portfolio-size or fall back to historical n_policies
    if args.portfolio_size is None and "n_policies" in meta:
        portfolio_size = int(meta["n_policies"])
    elif args.portfolio_size is not None:
        portfolio_size = args.portfolio_size
    else:
        raise RuntimeError("Provide --portfolio-size or include n_policies in DB meta")

    # ========== DISPLAY PARAMETERS ==========
    print(f"Estimated lambda_per_policy={lambda_per_policy:.6f}")
    if sev is None:
        print("No historical severities found; severity fit skipped.")
    else:
        print(f"Fitted severity: {sev}")

    # ========== RUN SIMULATIONS ==========
    # Execute Monte Carlo simulation
    sim_totals = run_simulation(lambda_per_policy, sev, portfolio_size, args.n_sims)
    
    # ========== COMPUTE RISK METRICS ==========
    # Calculate portfolio-level statistics: mean, volatility, VaR, percentiles
    summary = summarize_sim(sim_totals)
    
    # ========== DISPLAY RESULTS ==========
    print("Portfolio Risk Metrics:")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k}: {v:,.2f}")

    # ========== SAVE OUTPUTS ==========
    # Save all simulated loss values to CSV (for further analysis/visualization)
    pd.Series(sim_totals, name="total_loss").to_csv(args.out_csv, index=False)
    print(f"\nWrote {len(sim_totals)} simulation results to {args.out_csv}")


if __name__ == "__main__":
    # Run main simulation pipeline when script is executed directly
    main()
