#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# generate a small db (fast)
python src/create_db.py --out claims.db --n-policies 100 --n-years 1 --lambda-per-policy 0.05 --seed 1

# run a short monte carlo smoke test
python src/monte_carlo.py --db claims.db --n-sims 1000 --portfolio-size 100
echo "Done. See simulated_totals.csv"