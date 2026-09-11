# Training

Optimization output is written under `training/runs/` and is gitignored.

## Start a run

From the repo root:

```
PYTHONPATH=. python training/train.py training/example_plan.json
```

Copy `example_plan.json`, set `run_id`, `features`, and `runs`, then pass that path. The plan is copied into the run folder.

```
PYTHONPATH=. python training/train.py training/runs/<run_id>/run_details/plan.json
```

`stage: "adjs"` optimizes feature weights. `stage: "base"` runs the BaseHFA brute-force grid (long).

## plan.json

| Field | Default | Notes |
|---|---|---|
| `run_id` | required | Folder name under `training/runs/` |
| `stage` | `adjs` | `adjs` or `base` |
| `features` | names in `parameters.json` | Feature names to include. After adding a Feature class, add its name here |
| `runs` | `40000` | Number of optimizations (`adjs` only) |
| `hold_out` | `true` | 25% chance to drop one feature on an optimization |
| `base` | see example | `BaseHFA` parameters for this run. For `adjs`, default `level_weeks` is 15 |

SLSQP settings (`tol`, `step`, `method`) are on `Optimizer`. Each `adjs` optimization uses a random 60/40 split of games.

## Output

```
training/runs/{run_id}/
  optimization_records.csv    # adjs
  optimizer_results.csv       # base
  run_details/
    plan.json
    summary.json
```

`optimization_records.csv` columns: `optimization_time`, train/test RMSE for adj, base, and static, lifts, feature weights, `held_out_feature`, `hop`.

`summary.json` fields: `run_id`, `stage`, `runs`, `wall_seconds`, `hold_out`, `features`, `records`.

## Public API

`nfelohfa.optimize_adjs(...)` and `optimize_base()` write under `training/runs/` (folders `adjs` and `base`). `train.py` with a plan writes a named run folder that includes `run_details/plan.json` and `summary.json`.
