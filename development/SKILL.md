---
name: feature-exploration
description: End-to-end nfelohfa HFA feature exploration: research folder, residual analysis, Feature class + registry, 40k train.py runs, one-table metrics, dashboard, and a triangulated ship/no-ship call. Use when exploring, adding, testing, or recommending a new nfelohfa HFA feature.
---

# Feature exploration

Games are too volatile, SRS is imperfect, and we are looking for small amounts of lift. We will not find features that pass normal data science bars for validity. Create informed judgement calls by triangulating the observable data, research, justifiable mechanics via research, and the 40k sampling approach. Those should tell a consistent story. If they cannot, it is probably not a good feature.

Do not use p-values to close a question. 

## Environment
Feature exploration is...exploration. Not all feature explorations will be successful and therefore we do not want to opperate in the primary checkout until it is time to consolidate. The primary checkout is the repo the user has open for the package.

Create one worktree for the feature exploration, outside the primary checkout. Git cannot place a worktree inside the primary checkout, including inside `.local/`. Do not edit the primary checkout during exploration. Do not check out `main` in the worktree. Do not commit.

The primary checkout is the installed package. `import nfelohfa` loads the primary checkout unless the command runs from the worktree with `PYTHONPATH` set to the worktree. Run every command from the worktree root.

A worktree of `HEAD` does not contain uncommitted edits in the primary checkout. Start the worktree from the commit the exploration should run on.

Write the exploration folder inside the worktree. At cleanup, copy it into the primary checkout's `.local/` for user review, then remove the worktree.

## Folder

`.local/` is gitignored. Create this folder inside the worktree. Copy it into the primary checkout only at cleanup. One folder per feature, named as the registry name (`week_1`, not `Week 1`):

```
.local/feature-exploration/<feature>/
  research/          # lit review, copied memos
  analysis/          # residual scripts + json/csv
  plans/             # plan.json copies used to launch runs
  runs/              # copied training/runs/<run_id> output
  README.md          # definitions, cuts, pointer to FINDINGS
  FINDINGS.md        # numbers + recommendation
  IMPLEMENTATION.md  # instructions for a coding agent to implement the feature (ie feature definition, registry adding, dataloader utilities, etc)
```

Example: `.local/feature-exploration/week_1/`.

## 1. Research

Lit review and a first data cut so the signal has a mechanical story before it has a class.

- What component of HFA could this change, and why?
- What would falsify it?
- Does an existing feature already cover it (byes, surface, time, div)?
- Read memos already in the primary checkout's `.local/feature-exploration/`. Do not redo a memo that exists. Write new research in the worktree's exploration folder.

Observed HFA in this package is `mean(result - (home_rating - away_rating))` on `BaseHFA.prep_games` filters: played, REG, non-neutral, has ratings, drop 2020.

## 2. Implement

Match existing features. `apply_mode` is `multiply` unless there is a hard reason not to. Weight is a fraction of `hfa_base`.

```
nfelohfa/Model/Features/features/<name>.py
nfelohfa/Model/Features/features/__init__.py
nfelohfa/Model/Features/registry.py
```

These files are created in the worktree only. They are discarded when the worktree is removed. The spec that can be added to the primary checkout later is `IMPLEMENTATION.md` in the exploration folder.

DataLoader utility only if `build()` needs structure that is not already on the games frame (`week`, `div_game`, `home_previous_week`, surfaces, timezones). Pattern: `nfelohfa/Data/utilities/` + call from DataLoader. That utility is created in the worktree only, and discarded with it.

Do not add the name to `parameters.json`, `training/plan.py` `DEFAULT_FEATURES`, or `nfelohfa.optimize_adjs` defaults until you ship.

`Feature.apply` and `AdjustedHFA.apply_features` must stay unrounded during optimize. Round only with `round_output=True` in `calc_hfa`. If a run returns all weights at 0, the inner loop is rounding again.

## 3. Train

From the worktree root:

```
PYTHONPATH=. python training/train.py <plan.json>
```

Plan lives in the exploration `plans/` folder. `run_id` writes `training/runs/<run_id>/` (gitignored).

- Features list = existing five plus the new name.
- Scout (~8k) to confirm weights move and the sign is not noise.
- Then `runs: 40000`, `hold_out: true`, adj `base` as in `training/example_plan.json`.
- Iterate the spec (definition of the column), not the optimizer.
- Copy each finished run into `runs/` in the exploration folder.

## 4. Score

One table, every registered feature in the run:

| Column | Definition |
|---|---|
| % of games | share where the feature is on (`== 1`, or `!= 0` if continuous) |
| Conditional lift | RMSE with vs without that feature's adj, **on those games only** |
| Total lift | same comparison on **all** games |
| Median, p05, p95 | in-model 40k draws (not held out) |
| % < 0 | share of in-model draws below 0 |
| Straddle rate | mass on the minority side of 0: `min(%<0, 1-%<0)` |
| Mag. ratio | even vs odd `delta = mean(error\|on) - mean(error\|off)`; `min(\|e\|,\|o\|)/max(\|e\|,\|o\|)` |

Weights at 40k medians for the lift columns. Error is `result - (home_rating - away_rating)`.

Primary: 40k interval vs 0, and whether conditional lift is in the same band as shipped features. Mag. ratio is secondary. Full-sample total lift will look tiny on a rare flag; that is base rate, not a veto.

Write numbers into `FINDINGS.md`. Recommendation is ship / do not ship, with the weight if ship, and whether the four legs agree.

## 5. Dashboard

One page. Headline, recommendation, and the table from section 4. No second weights table. No second stability table.

- Stats row: space-between, full width.
- Weight distribution and test-lift distribution: area charts, 0 at the center, vertical line at 0. Strong stroke, fill ~16% opacity.
- Feature / adjusted model in the accent color. Rolling base and other reference series muted.
- Do not overload. No leftover week-by-week walls once the table exists.

Copy [templates/dashboard.canvas.tsx](templates/dashboard.canvas.tsx). Filled example: [examples/week-1-hfa-feature.canvas.tsx](examples/week-1-hfa-feature.canvas.tsx). Write the dashboard in the worktree's `.local/feature-exploration/<feature>/`.

## 6. Cleanup

Complete FINDINGS.md and IMPLEMENTATION.md. Implementation should only be scoped to the spec that would be added to the codebase should the user approve. Failed alternative specs can stay in the exploration folder, but they should not be added as new features to the codebase via implimentation.

Copy `.local/feature-exploration/<feature>/` from the worktree into the primary checkout's `.local/feature-exploration/<feature>/` before removing anything. `training/runs/` in the worktree is deleted with the worktree, so each finished run must already be in the exploration folder's `runs/`.

Then remove the worktree. If a branch was created for it, delete that branch too.

The feature module, registry edit, DataLoader utility, and any other package edits in the worktree are not copied back. `IMPLEMENTATION.md` is the handoff.

## IMPORTANT
- Do not wrap lines manually when generating markdown under any circumstance. The user's markdown viewer will handle warpping.
- Do not use p-values to close a question.