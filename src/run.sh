#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pip3 install -q -r requirements.txt

python3 src/create_db.py --out claims.db --n-policies 300 --n-years 3 --lambda-per-policy 0.1 --seed 1
python3 src/monte_carlo.py --db claims.db --n-sims 1000 --seed 1
echo "Done. See simulated_totals.csv"