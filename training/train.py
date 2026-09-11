'''
Launch a training run from a plan.json.

From the repo root:
    PYTHONPATH=. python training/train.py training/example_plan.json
    PYTHONPATH=. python training/train.py training/runs/<run_id>/run_details/plan.json
'''
import json
import pathlib
import sys
import time

TRAIN_DIR = pathlib.Path(__file__).parent.resolve()
REPO_ROOT = TRAIN_DIR.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nfelohfa.Optimizer import Optimizer
from training.plan import Plan


def write_summary(plan, wall_seconds, n_runs, records_name):
    '''
    Writes run_details/summary.json
    '''
    summary = {
        'run_id' : plan.run_id,
        'stage' : plan.stage,
        'runs' : n_runs,
        'wall_seconds' : wall_seconds,
        'hold_out' : plan.hold_out,
        'features' : plan.features,
        'records' : records_name
    }
    out = plan.details_dir / 'summary.json'
    with open(out, 'w') as fp:
        json.dump(summary, fp, indent=4)
    return out


def run_training(plan):
    '''
    Write plan.json, run optimization, write records at the run root
    and summary.json under run_details/.
    '''
    plan.details_dir.mkdir(parents=True, exist_ok=True)
    plan.save(plan.details_dir / 'plan.json')
    started = time.time()
    if plan.stage == 'base':
        Optimizer.optimize_base(
            output_dir=str(plan.run_dir),
            level=plan.base['level'],
            level_weeks=plan.base['level_weeks'],
            reg_weeks=plan.base['reg_weeks'],
            kick_in=plan.base['kick_in']
        )
        n_runs = None
        records_name = 'optimizer_results.csv'
    else:
        n_runs = Optimizer.optimize_adjs(
            features=plan.features,
            runs=plan.runs,
            hold_out=plan.hold_out,
            output_dir=str(plan.run_dir),
            level=plan.base['level'],
            level_weeks=plan.base['level_weeks'],
            reg_weeks=plan.base['reg_weeks'],
            kick_in=plan.base['kick_in']
        )
        records_name = 'optimization_records.csv'
    wall_seconds = time.time() - started
    write_summary(plan, wall_seconds, n_runs, records_name)
    return plan.run_dir


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python training/train.py <plan.json>')
        sys.exit(1)
    plan = Plan.load(sys.argv[1])
    run_dir = run_training(plan)
    print('Run written to {0}'.format(run_dir))
