---
name: feature-exploration
description: End-to-end nfelohfa HFA feature exploration: research folder, residual analysis, Feature class + registry, 40k train.py runs, one-table metrics, dashboard, and a triangulated ship/no-ship call. Use when exploring, adding, testing, or recommending a new nfelohfa HFA feature.
---

# Feature exploration

Games are too volatile, SRS is imperfect, and we are looking for small amounts of lift. We will not find features that pass normal data science bars for validity. Create informed judgement calls by triangulating the observable data, research, justifiable mechanics via research, and the 40k sampling approach. Those should tell a consistent story. If they cannot, it is probably not a good feature.

Do not use p-values to close a question. 

## Environment
Feature exploration is...exploration. Not all feature explorations will be successful and therefore we do not want to opperate in the primary checkout until it is time to consolidate. The primary checkout is the repo the user has open for the package.

Create one worktree for the feature exploration, outside the primary checkout. Git cannot place a worktree inside the primary checkout, including inside `.local/`. Do not edit the primary checkout during exploration. Do not check out `main` in the worktree. Do not commit. Here, "one worktree for feature exploration" means a worktree for the session, even if multiple features are being explored. It does not mean one worktree per feature UNLESS multiple features are being explored in parallel. To avoid collisions in that case, multiple worktrees would be needed.

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

### Conceptionalization

Features should reflect the mechanics of the research and availability of the data. They are a model of the dynamic in code. 

They may be simple or complex. They can utilize binary flags and gates, or be continuous. What is important is that they capture the ethos and mechanics represented by the research. If this feature is a net new discovery and has no public research available, it should match the hypothesis.

For example,
* It is documented that teams play better after a bye week. The feature is a simple yes/no flag for if the team is coming off a bye
* Research shows circadian rhythms have a continuous effect on performance, with a peak occuring in the afternoon. The feature should be continuous and reflect team's net distance from their respective peak performance window

To explore the possible space for the feature, you can take shortcuts / rough cuts by examining the margin residual when a feature is on vs off. Said another way, if there is a large difference in actual vs predict margin when the feautre is active vs inactive, that is a good signal that the feature is capturing an effect. To benchmark, here are on/off differences for some existing validated features:

On is `== 1` and off is `== 0`. Home time advantage is continuous, so on is `> 0` (home closer to its peak than the away team) and off is `<= 0`. Each value is `mean(result - (home_rating - away_rating))` in points on `BaseHFA.prep_games` (regular season, non-neutral, has ratings, 2020 dropped; null results dropped from the mean), seasons 1999–2026. Difference is on minus off.

| Feature | HFA on | HFA off | On/Off Difference |
|---|---|---|---|
| Home bye | 2.855 | 2.307 | +0.548 |
| Away bye | 0.585 | 2.453 | -1.868 |
| Home time advantage | 2.870 | 2.046 | +0.824 |
| Different surface | 2.648 | 2.022 | +0.626 |
| Division game | 1.911 | 2.614 | -0.703 |
| Week 1 | 1.091 | 2.426 | -1.335 |


Degrees of freedom should also be considered for feature viability. Simple features are more likely to represent repeatable signal vs heavily sliced and contorted ones. For instance, a simple feature might be just is_bye. A more complex one might be is_home_bye and is_away_bye. The added complexity IS supported by research, mechanics, and the observed data. However splitting it futher to features like is_home_bye_after_week_10 then enters the territory of tortured and overfit.

### Structure

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
- Scout (~8k) to confirm weights move and the sign is not noise (< 10% of runs crossing 0).
-- If the scout does not pass (>10% of runs crossing 0), or passes weakly (>5% of runs crossing 0), iterate on the feature (conceptionalization step) and scout again.
-- Continue this process until a viable, quality candidate is found OR until you've exhausted all reasonable routes to express the idea.
-- If no reasonable candidate exists, skip the larger 40000 run
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