
# Monte Carlo Claims Model (SQL + Python)

This project is a Monte Carlo simulator that models yearly insurance losses across a segmented portfolio of policies (home, auto, commercial). It uses a Poisson model to estimate claim frequency and a fitted log-normal severity distribution to estimate claim size — fitted separately per segment from historical data stored in SQLite. By running thousands of simulations, the program generates a full portfolio loss distribution and risk metrics including VaR. Each simulation run is persisted back to the database for comparison across scenarios.

**Contents**
- `requirements.txt`: Python dependencies
- `src/create_db.py`: generates `claims.db` with synthetic segmented claims history
- `src/monte_carlo.py`: fits frequency/severity per segment, runs Monte Carlo sims, saves results to DB
- `src/run.sh`: convenience runner (installs deps and runs a smoke test)

**Quick Start (macOS / zsh)**

Install dependencies and run a smoke test:

```bash
pip3 install -r requirements.txt
python3 src/create_db.py --out claims.db --n-policies 300 --n-years 3 --lambda-per-policy 0.1 --seed 1
python3 src/monte_carlo.py --db claims.db --n-sims 5000 --seed 42
```

Or use the included helper:

```bash
bash src/run.sh
```

The `monte_carlo.py` script prints per-segment fitted parameters, summary risk metrics (mean, volatility, VaR), and saves results to both `claims.db` and `simulated_totals.csv`.

**How SQL is used**

The database has five tables:

| Table | Purpose |
|---|---|
| `policies(policy_id, segment)` | Maps each policy to its segment (home / auto / commercial) |
| `claims(id, policy_id, claim_date, amount)` | Historical claim records with FK to policies |
| `meta(key, value)` | Portfolio metadata (n_policies, n_years, base_lambda) |
| `simulation_runs(id, created_at, seed, n_sims, portfolio_size, mean_loss, ...)` | One row per simulation run with summary metrics |
| `simulation_results(id, run_id, total_loss)` | Every simulated loss value, linked to its run |

SQL does meaningful work in three places:

1. **Per-segment policy counts** — `COUNT(*) ... GROUP BY segment` on the `policies` table
2. **Per-segment claim counts** — `LEFT JOIN claims ON policies` + `GROUP BY segment` to compute lambda per segment
3. **Per-segment claim amounts** — `JOIN` between `claims` and `policies` to pull amounts per segment for log-normal fitting
4. **Storing simulation outputs** — each run is written back to `simulation_runs` and `simulation_results`, so you can query and compare runs across different seeds or portfolio sizes

**Policy segments**

Policies are split evenly across three segments, each with different risk profiles:

| Segment | Lambda multiplier | Severity |
|---|---|---|
| home | 1.0× base | Medium frequency, heavy tail |
| auto | 1.5× base | Higher frequency, smaller claims |
| commercial | 0.5× base | Lower frequency, very large claims |

Frequency and severity are fitted independently per segment from historical data using SQL GROUP BY queries.

**Viewing the DB**
- Use the VS Code SQLite extension or DB Browser for SQLite to open `claims.db`
- Quick CLI preview:

```bash
sqlite3 claims.db ".schema"
sqlite3 -header -csv claims.db "SELECT * FROM claims LIMIT 20;"
```

Query simulation run history:

```bash
sqlite3 -header claims.db "SELECT id, created_at, seed, n_sims, mean_loss, var_95 FROM simulation_runs;"
```

## Playing With the Inputs

### 1. Change the number of policies

More policies → more expected claims → higher aggregate losses.
```bash
python3 src/create_db.py --out claims.db --n-policies 2000 --n-years 10 --lambda-per-policy 0.1 --seed 1
```

### 2. Adjust the base claim frequency (λ)

`--lambda-per-policy` is the base rate, scaled per segment (auto gets 1.5×, commercial gets 0.5×).

**Low frequency (rare events):**
```bash
--lambda-per-policy 0.02
```

**High frequency (many more claims):**
```bash
--lambda-per-policy 0.2
```

### 3. Change the number of simulations

More simulations → smoother and more reliable distribution estimates.
```bash
python3 src/monte_carlo.py --db claims.db --n-sims 10000 --seed 42
```

- **Low `n_sims`** (e.g., 500): fast but noisy
- **High `n_sims`** (e.g., 10000+): slower but more stable tail metrics

### 4. Vary the portfolio size

`--portfolio-size` sets the total policies to simulate. Each segment's share is scaled proportionally to its historical share.
```bash
python3 src/monte_carlo.py --db claims.db --portfolio-size 5000 --seed 42
```

### 5. Try stress-test scenarios

#### A. High frequency across all segments
```bash
python3 src/create_db.py --out claims.db --n-policies 1000 --n-years 10 --lambda-per-policy 0.3 --seed 5
python3 src/monte_carlo.py --db claims.db --n-sims 5000 --seed 5
```

#### B. Compare two runs across seeds
```bash
python3 src/monte_carlo.py --db claims.db --n-sims 5000 --seed 1
python3 src/monte_carlo.py --db claims.db --n-sims 5000 --seed 99
sqlite3 -header claims.db "SELECT id, seed, mean_loss, var_95 FROM simulation_runs;"
```

### What to look for while experimenting

- **Segment differences** → auto drives frequency, commercial drives tail severity
- **Mean vs Median** → right-skew from log-normal severity
- **VaR 95 / 99** → behaviour of the extreme tail
- **Portfolio size effects** → diversification benefit as n_policies grows
- **Run history** → compare mean_loss and var_95 across runs in simulation_runs table

Playing with these inputs builds intuition for how actuaries explore uncertainty, test pricing assumptions, and understand capital needs.
