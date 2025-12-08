# Monte Carlo Claims Model (SQL + Python)

This project demonstrates a simple Monte Carlo claims model using a SQL-backed historical dataset (SQLite) and Python for parameter estimation and simulation.

Contents
- `requirements.txt` — Python dependencies
- `src/create_db.py` — creates `claims.db` with synthetic historical data
- `src/monte_carlo.py` — fits frequency/severity and runs Monte Carlo simulations

Quick start

1. Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Generate a sample SQLite DB:

```bash
python src/create_db.py --out claims.db
```

3. Run the Monte Carlo model (default 10k sims):

```bash
python src/monte_carlo.py --db claims.db --n-sims 10000
```

The script prints summary statistics (mean, std, VaR, CVaR) and writes `simulated_totals.csv` with simulation results.

Customization
- Use `--portfolio-size` to simulate different portfolio sizes.
- Use `--n-sims` to increase/decrease simulation count.

Notes
- This is a small, focused example. Replace the synthetic DB with your production SQL database and adjust distribution choices as appropriate.
# Monte-Carlo-Simulation
