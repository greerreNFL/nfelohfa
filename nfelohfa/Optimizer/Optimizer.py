import pandas as pd
import numpy
import pathlib
import random
import time

from scipy.optimize import minimize

from ..Data import DataLoader
from ..Model import BaseHFA, AdjustedHFA
from ..Model.Features import build_all
from ..Grading import Grader


class Optimizer:
    '''
    Fits BaseHFA params (brute force) and feature weights (SLSQP).
    Train/test split and SLSQP settings match the previous AdjustedHFA.optimize.
    '''
    def __init__(self, games, features, level=2.5, tol=0.000001, step=0.00001, method='SLSQP'):
        self.games = games.copy()
        self.features = features
        self.features_key_array = list(features.keys()) ## keep ordering for reference
        self.features_value_array = [self.features[k] for k in self.features_key_array]
        self.level = level
        self.adj = AdjustedHFA(games=self.games, features=self.features)
        self.grader = Grader(level=level)
        ## optimizer ##
        self.train_df, self.test_df = self.train_test_split()
        self.best_guesses = [v/2 + .5 for v in self.features_value_array] ## normalized
        self.bounds = tuple((0, 1) for _ in range(len(self.best_guesses))) ## normalized
        self.optimized_features = None
        self.optimization_record = {}
        self.tol = tol
        self.step = step
        self.method = method

    def train_test_split(self):
        '''
        Randomly split the df into a 60/40 train/test
        '''
        mask = numpy.random.choice(
            a=['train', 'test'],
            size=len(self.games),
            p=[.6,.4]
        )
        return self.games[mask=='train'].copy(), self.games[mask=='test'].copy()

    def gen_feature_dict(self, x):
        '''
        Takes a normalized array of values from the optimzer, "x", and translates
        in to a dictionary of features
        '''
        features = {}
        for i,v in enumerate(x):
            features[self.features_key_array[i]] = (v - 0.5) * 2 ## denorm
        ## return ##
        return features

    def get_rmses(self, x, df):
        '''
        wrapper to gen a feature dict, apply to df, and calc rmses
        '''
        ## get denormed feature dict ##
        features = self.gen_feature_dict(x)
        ## apply to df ##
        applied_df = self.adj.apply_features(df=df,features=features)
        ## calc rmses
        rmse_adj, rmse_base, rmse_static = self.grader.calc_rmse(applied_df)
        ## return ##
        return rmse_adj, rmse_base, rmse_static

    def obj_func(self, x, train_df):
        '''
        Objective function for the optimizer
        '''
        rmse_adj, rmse_base, rmse_static = self.get_rmses(x, train_df)
        ## return ##
        return rmse_adj

    def optimize(self):
        '''
        Run an optimization over the training set
        '''
        ## optimize ##
        opti_time_start = float(time.time())
        solution = minimize(
                self.obj_func,
                self.best_guesses,
                args=(self.train_df),
                bounds=self.bounds,
                method=self.method,
                options={
                    'ftol' : self.tol,
                    'eps' : self.step
                }
            )
        opti_time_end = float(time.time())
        ## get the result ##
        self.optimized_features = self.gen_feature_dict(solution.x)
        train_rmse_adj, train_rmse_base, train_rmse_static = self.get_rmses(solution.x, self.train_df)
        test_rmse_adj, test_rmse_base, test_rmse_static = self.get_rmses(solution.x, self.test_df)
        ## create the record ##
        self.optimization_record = {
            'optimization_time' : opti_time_end - opti_time_start,
            'train_rmse_adj' : train_rmse_adj,
            'train_rmse_base' : train_rmse_base,
            'train_rmse_static' : train_rmse_static,
            'test_rmse_adj' : test_rmse_adj,
            'test_rmse_base' : test_rmse_base,
            'test_rmse_static' : test_rmse_static,
            'train_lift_adj' : train_rmse_static / train_rmse_adj - 1,
            'train_lift_base' : train_rmse_static / train_rmse_base - 1,
            'test_lift_adj' : test_rmse_static / test_rmse_adj - 1,
            'test_lift_base' : test_rmse_static / test_rmse_base - 1
        }
        ## add feature values ##
        for k,v in self.optimized_features.items():
            self.optimization_record[k] = v

    @staticmethod
    def optimize_base(
            output_dir=None,
            level=2.5,
            level_weeks=75,
            reg_weeks=240,
            kick_in=.75
        ):
        '''
        Runs the base optimization, which is just a brute force
        Output is saved to output_dir (training run folder)
        '''
        dl = DataLoader()
        base = BaseHFA(
            dl.db['games'],
            level_weeks,
            reg_weeks,
            level,
            kick_in
        )
        if output_dir is not None:
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            base.folder_loc = pathlib.Path(output_dir).resolve()
        result = base.optimize()
        ## 
        return result

    @staticmethod
    def optimize_adjs(
            features=[
                'home_bye', 'away_bye', 'home_time_advantage',
                'dif_surface', 'div_game'
            ],
            runs=40000,
            hold_out=True,
            output_dir=None,
            level=2.5,
            level_weeks=15,
            reg_weeks=280,
            kick_in=.75
        ):
        '''
        Randomly samples
        the games since 1999 a set number of times and calculates
        the optimal adjs for a set of HFA features
        * features: an array with the features to optimize
        * runs: number of times to perform the optimization
        * hold_out: if set to true, randomly holds out one feature
        Output is saved to output_dir / optimization_records.csv
        '''
        print('Optimizing the HFA Adj parameters...')
        ## load data ##
        dl = DataLoader()
        base = BaseHFA(
            dl.db['games'],
            level_weeks,
            reg_weeks,
            level,
            kick_in
        )
        ## train on the BaseHFA.prep_games population ##
        games = base.prep_games(build_all(base.games_w_hfa))
        ## struc to save results ##
        package_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
        if output_dir is None:
            output_dir = '{0}/training/runs/adjs'.format(package_dir)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        save_loc = '{0}/optimization_records.csv'.format(output_dir)
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
                temp.to_csv(save_loc, index=False)
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
            opt = Optimizer(
                games=games,
                features=features_local,
                level=level
            )
            opt.optimize()
            ## index in optimization_records.csv ##
            opt.optimization_record['hop'] = run
            ## if a feature was held out, add its absence to the record ##
            if hold_out_feature:
                opt.optimization_record[hold_out_feature] = numpy.nan
                opt.optimization_record['held_out_feature'] = hold_out_feature
            else:
                opt.optimization_record['held_out_feature'] = numpy.nan
                ## update features dict for speed in next optimization ##
                ## but only if it was run with the complete set
                features_dict = opt.optimized_features
            ## append rec ##
            optimization_records.append(opt.optimization_record)
        ## final save (including when runs is not a multiple of 1000) ##
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(optimization_records).to_csv(save_loc, index=False)
        return len(optimization_records)
