
# Monte Carlo Claims Model (SQL + Python)

This project is a Monte Carlo simulator that models yearly insurance losses for a portfolio of policies. It uses a Poisson model to estimate how many claims happen and a fitted severity distribution (like Gamma) to estimate how large those claims are. By running thousands of simulations, the program shows how total losses can vary from year to year and helps visualise both the “typical” outcomes and the rare, more extreme ones. The goal is to give an intuitive understanding of how actuaries combine frequency severity models, aggregate losses, and risk measures like VaR and CVaR to assess portfolio risk and make decisions around pricing, capital, and reinsurance.

**Contents**
- `requirements.txt`: Python dependencies
- `src/create_db.py`: generator to create a sample `claims.db` (SQLite)
- `src/monte_carlo.py`: loads data, fits frequency/severity, runs Monte Carlo sims
- `sample_claims.csv`: small sample export (for quick inspection)
- `run.sh`: convenience runner (creates venv, installs deps, runs a smoke test)

**Quick Start (macOS / zsh)**
- Clone the repo and move into it (quote path if it contains spaces):

```bash
cd '/Users/mehul/Github Projects/monte_carlo_claims'
```

- Create and activate a virtual environment, install dependencies, generate a small DB and run a short smoke test:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/create_db.py --out claims.db --n-policies 100 --n-years 1 --lambda-per-policy 0.05 --seed 1
python src/monte_carlo.py --db claims.db --n-sims 1000 --portfolio-size 100
```

- Or use the included helper (handles quoting and venv setup):

```bash
./run.sh
```

The `monte_carlo.py` script prints summary statistics (mean, std, VaR, CVaR) and writes `simulated_totals.csv` with simulation results.

**How SQL is used**
- I used SQLite (`claims.db`) as the portable store for historical claims.
- `create_db.py` creates two tables: `claims(id, policy_id, claim_date, amount)` and `meta(key,value)`.
- `monte_carlo.py` reads those tables with `pandas.read_sql_query`, then fits distributions and runs simulations in memory.

**Viewing the DB**
- Use the VS Code SQLite extension or DB Browser for SQLite to open `claims.db`
- Quick CLI preview (no GUI):

```bash
sqlite3 claims.db ".schema"
sqlite3 -header -csv claims.db "SELECT * FROM claims LIMIT 20;"
```

You can export a small CSV for sharing:

```bash
sqlite3 -header -csv claims.db "SELECT * FROM claims LIMIT 500;" > sample_claims.csv
```

## Playing With the Inputs

You can change how the simulator behaves by tweaking the inputs passed into `create_db.py` and `monte_carlo.py`. This lets you explore how different assumptions affect claim frequency, claim severity, and the overall loss distribution.

### 1. Change the number of policies

More policies → more expected claims → higher aggregate losses.
```bash
python src/create_db.py --out claims.db --n-policies 2000 --n-years 10 --lambda-per-policy 0.05 --seed 1
```

### 2. Adjust the claim frequency (λ)

`--lambda-per-policy` controls how often each policy generates claims per year.

**Low frequency (rare events):**
```bash
--lambda-per-policy 0.02
```

**High frequency (many more claims):**
```bash
--lambda-per-policy 0.2
```

λ has one of the biggest impacts on total losses — higher λ means more claims, which pushes the entire loss distribution upward.

### 3. Change the number of simulations

More simulations → smoother and more reliable distribution estimates.
```bash
python src/monte_carlo.py --db claims.db --n-sims 5000 --portfolio-size 100
```

- **Low `n_sims`** (e.g., 500): fast but noisy
- **High `n_sims`** (e.g., 5000+): slower but more stable tail metrics (VaR, CVaR)

### 4. Vary the portfolio size

`--portfolio-size` is how many policies you model during the simulation, not in the database.
```bash
python src/monte_carlo.py --db claims.db --portfolio-size 500
```

**Effects:**
- Large portfolios → smoother results (law of large numbers)
- Small portfolios → more volatility and more extreme outcomes

### 5. Change the random seed

Different seeds produce different synthetic claim histories.
```bash
--seed 42
```

**Use this when:**
- you want reproducible results
- you're exploring sensitivity to stochastic variation

### 6. Try stress-test scenarios

#### A. High frequency + heavy severity
Simulates a rough underwriting year with many claims:
```bash
python src/create_db.py --out claims.db --n-policies 1000 --n-years 10 --lambda-per-policy 0.2 --seed 5
python src/monte_carlo.py --db claims.db --n-sims 2000 --portfolio-size 1000
```

#### B. Rare but very costly claims
Frequency is low, but severity drives risk:
```bash
python src/create_db.py --out claims.db --n-policies 1000 --n-years 10 --lambda-per-policy 0.01 --seed 7
```

Use this to see how "long-tail" risk shows up in VaR and CVaR metrics.

### What to look for while experimenting

- **Mean vs Median** → shows right-skew from severity
- **VaR / CVaR** → behaviour of the extreme tail
- **Standard deviation** → overall volatility
- **Portfolio size effects** → diversification benefit
- **Sensitivity to λ** → how claim frequency shifts the entire risk profile

Playing with these inputs builds intuition for how actuaries explore uncertainty, test pricing assumptions, and understand capital needs.

