import json
import pathlib


REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
RUNS_DIR = REPO_ROOT / 'training' / 'runs'

DEFAULT_FEATURES = [
    'home_bye', 'away_bye', 'home_time_advantage',
    'dif_surface', 'div_game'
]
## adj defaults; not loaded from parameters.json ##
DEFAULT_ADJ_BASE = {
    'level' : 2.5,
    'level_weeks' : 15,
    'reg_weeks' : 280,
    'kick_in' : 0.75
}
## base brute force uses these starting knobs ##
DEFAULT_BASE_START = {
    'level' : 2.5,
    'level_weeks' : 75,
    'reg_weeks' : 240,
    'kick_in' : 0.75
}


class Plan:
    '''
    One training run. Written to run_details/plan.json.
    '''
    def __init__(
            self,
            run_id,
            stage='adjs',
            features=None,
            runs=40000,
            hold_out=True,
            base=None
        ):
        self.run_id = run_id
        self.stage = stage
        self.features = list(features) if features is not None else list(DEFAULT_FEATURES)
        self.runs = runs
        self.hold_out = hold_out
        if base is not None:
            self.base = dict(base)
        elif stage == 'base':
            self.base = dict(DEFAULT_BASE_START)
        else:
            self.base = dict(DEFAULT_ADJ_BASE)

    @property
    def run_dir(self):
        return RUNS_DIR / self.run_id

    @property
    def details_dir(self):
        return self.run_dir / 'run_details'

    def to_dict(self):
        return {
            'run_id' : self.run_id,
            'stage' : self.stage,
            'features' : self.features,
            'runs' : self.runs,
            'hold_out' : self.hold_out,
            'base' : self.base
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            run_id=data['run_id'],
            stage=data.get('stage', 'adjs'),
            features=data.get('features'),
            runs=data.get('runs', 40000),
            hold_out=data.get('hold_out', True),
            base=data.get('base')
        )

    @classmethod
    def load(cls, path):
        with open(path) as fp:
            return cls.from_dict(json.load(fp))

    def save(self, path=None):
        if path is None:
            path = self.details_dir / 'plan.json'
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as fp:
            json.dump(self.to_dict(), fp, indent=4)
        return path
