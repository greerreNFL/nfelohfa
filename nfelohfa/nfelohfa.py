import pandas as pd
import numpy
import pathlib
import random
import json

from .DataLoader import DataLoader
from .BaseHFA import BaseHFA
from .AdjustedHFA import AdjustedHFA

## package_dir ##
package_dir = pathlib.Path(__file__).parent.parent.resolve()
## load params ##
with open('{0}/parameters.json'.format(package_dir)) as fp:
    config = json.load(fp)

def calc_hfa():
    '''
    Calcualtes HFA estimates by game and saves to package
    '''
    dl = DataLoader()
    base = BaseHFA(
        games=dl.db['games'],
        level=config['base']['level'],
        level_weeks=config['base']['level_weeks'],
        reg_weeks=config['base']['reg_weeks'],
        kick_in=config['base']['kick_in']
    )
    adj = AdjustedHFA(
        games=base.games_w_hfa,
        features=config['features']
    )
    ## run ##
    hfa = adj.apply_features()
    ## save the output ##
    ## form cols ##
    cols = ['game_id', 'season', 'week', 'home_team', 'away_team']
    ## add the features ##
    for k,v in config['features'].items():
        cols.append(k)
    ## add base hfa ##
    cols.append('hfa_base')
    ## add adjustedments ##
    for k,v in config['features'].items():
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
    '''
    dl = DataLoader()
    base = BaseHFA(
        dl.db['games'],
        75,
        240,
        2.5,
        .75
    )
    result = base.optimize()
    ## 
    return result


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
    Output is saved to the AdjustedHFA folder of the package
    '''
    print('Optimizing the HFA Adj parameters...')
    ## load data ##
    dl = DataLoader()
    base = BaseHFA(
        dl.db['games'],
        15,
        280,
        2.5,
        .75
    )
    ## struc to save results ##
    save_loc = '{0}/AdjustedHFA/optimization_records.csv'.format(
        pathlib.Path(__file__).parent.resolve()
    )
    optimization_records=[]
    ## formulate features ##
    features_dict = {}
    for feature in features:
        features_dict[feature] = 0
    ## loop ##
    for run in range(0,runs):
        ## save output every 1k rounds
        if (run +1) % 1000 == 0:
            print('     On run {0} of {1}'.format(
                run+1,
                runs
            ))
            temp = pd.DataFrame(optimization_records)
            temp.to_csv(save_loc)
        ## make a copy of the features for a random removall ##
        features_local = features_dict.copy()
        hold_out_feature = None
        if hold_out:
            if random.random() < .25: ## 25% chance to hold out
                hold_out_feature = random.choice(
                    list(features_local.keys())
                )
                del features_local[hold_out_feature]
        ## run ##
        adj = AdjustedHFA(
            games=base.games_w_hfa,
            features=features_local,
            level=2.5
        )
        adj.optimize()
        ## if a feature was held out, add its absence to the record ##
        if hold_out_feature:
            adj.optimization_record[hold_out_feature] = numpy.nan
            adj.optimization_record['held_out_feature'] = hold_out_feature
        else:
            adj.optimization_record['held_out_feature'] = numpy.nan
            ## update features dict for speed in next optimization ##
            ## but only if it was run with the complete set
            features_dict = adj.optimized_features
        ## append rec ##
        optimization_records.append(adj.optimization_record)
    
