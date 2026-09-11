import pathlib

from .Data import DataLoader
from .Model import BaseHFA, AdjustedHFA
from .Model.Features import build_all
from .Optimizer import Optimizer
from .config import Config

## package_dir ##
package_dir = pathlib.Path(__file__).parent.parent.resolve()
## load params ##
config = Config.load()

def calc_hfa():
    '''
    Calcualtes HFA estimates by game and saves to package
    '''
    dl = DataLoader()
    base = BaseHFA(
        games=dl.db['games'],
        level=config.base['level'],
        level_weeks=config.base['level_weeks'],
        reg_weeks=config.base['reg_weeks'],
        kick_in=config.base['kick_in']
    )
    games = build_all(base.games_w_hfa)
    adj = AdjustedHFA(
        games=games,
        features=config.features
    )
    ## run ##
    hfa = adj.apply_features()
    ## save the output ##
    ## form cols ##
    cols = [
        'game_id', 'season', 'week', 'home_team', 'away_team',
        'gametime', 'stadium', 'location',
        'roof', 'surface', 'temp', 'wind'
    ]
    ## add the features ##
    for k,v in config.features.items():
        cols.append(k)
    ## add base hfa ##
    cols.append('hfa_base')
    ## add adjustedments ##
    for k,v in config.features.items():
        cols.append('{0}_adj'.format(k))
    ## add final ##
    cols.append('hfa_adj')
    ## save ##
    hfa[cols].to_csv(
        '{0}/estimated_hfa.csv'.format(package_dir),
        index=False
    )

def optimize_base():
    '''
    Runs the base optimization, which is just a brute force
    Output is saved to training/runs. Named runs with plan.json: training/train.py.
    '''
    return Optimizer.optimize_base()


def optimize_adjs(
        features=[
            'home_bye', 'away_bye', 'home_time_advantage',
            'dif_surface', 'div_game'
        ],
        runs=40000,
        hold_out=True
    ):
    '''
    Randomly samples
    the games since 1999 a set number of times and calculates
    the optimal adjs for a set of HFA features
    * features: an array with the features to optimize
    * runs: number of times to perform the optimization
    * hold_out: if set to true, randomly holds out one feature
    Output is saved to training/runs. Named runs with plan.json: training/train.py.
    '''
    return Optimizer.optimize_adjs(
        features=features,
        runs=runs,
        hold_out=hold_out
    )
