import pandas as pd
import numpy

from scipy.optimize import minimize
import time

class AdjustedHFA:
    '''
    Class for applying and training aditional features
    that adjust BaseHFA
    '''
    def __init__(self, games, features, level=2.5, tol=0.000001, step=0.00001, method='SLSQP'):
        self.games = games.copy()
        self.features = features
        self.features_key_array = list(features.keys()) ## keep ordering for reference
        self.features_value_array = [self.features[k] for k in self.features_key_array]
        self.level = level
        ## optimizer ##
        self.train_df, self.test_df = self.train_test_split()
        self.best_guesses = [v/2 + .5 for v in self.features_value_array] ## normalized
        self.bounds = tuple((0, 1) for _ in range(len(self.best_guesses))) ## normalized
        self.optimized_features = None
        self.optimization_record = {}
        self.tol = tol
        self.step = step
        self.method = method


    def apply_features(self, df=None, features=None):
        '''
        Applies features to the base hfa value
        '''
        ## if no features are passed, use the ones passed on init
        ## this is done so other features can be passed
        ## to the func with teh optimizer
        if features is None:
            features=self.features
        if df is None:
            df=self.games
        ## init hfa_adj ##
        df['hfa_adj'] = numpy.round(df['hfa_base'],3)
        for k,v in features.items():
            ## calc adj
            df['{0}_adj'.format(k)] = numpy.round((
                df['hfa_base'] *
                (df[k] * v) ## if not active, games[k] will be 0
            ),3)
            ## add to hfa_adj ##
            df['hfa_adj'] = numpy.round(df['hfa_adj'] + df['{0}_adj'.format(k)],3)
        ## if the field is neutral, remove the base, leaving only the adjs, which we
        ## hypothesize to still be relevant ##
        df['hfa_adj'] = numpy.where(
            df['location'] == 'Neutral',
            df['hfa_adj'] - df['hfa_base'],
            df['hfa_adj']
        )
        ## then also zero out the base for clarity ##
        df['hfa_base'] = numpy.where(
            df['location'] == 'Neutral',
            0,
            df['hfa_base']
        )
        ## return ##
        return df

    ## OPTIMIZER ##
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
    
    def calc_rmse(self, applied_df):
        '''
        Calc the rmse on a df that has had the adjs applied
        '''
        applied_df['proj_adj'] = (
            applied_df['home_team_rating'] +
            applied_df['hfa_adj'] -
            applied_df['away_team_rating']
        )
        applied_df['proj_base'] = (
            applied_df['home_team_rating'] +
            applied_df['hfa_base'] -
            applied_df['away_team_rating']
        )
        applied_df['proj_static'] = (
            applied_df['home_team_rating'] +
            self.level -
            applied_df['away_team_rating']
        )
        ## calc errors ##
        applied_df['se_adj'] = (
            applied_df['result'] - applied_df['proj_adj']
        ) ** 2
        applied_df['se_base'] = (
            applied_df['result'] - applied_df['proj_base']
        ) ** 2
        applied_df['se_static'] = (
            applied_df['result'] - applied_df['proj_static']
        ) ** 2
        ## return rmses
        return (
            applied_df['se_adj'].mean() ** (1/2),
            applied_df['se_base'].mean() ** (1/2),
            applied_df['se_static'].mean() ** (1/2)
        )

    def get_rmses(self, x, df):
        '''
        wrapper to gen a feature dict, apply to df, and calc rmses
        '''
        ## get denormed feature dict ##
        features = self.gen_feature_dict(x)
        ## apply to df ##
        applied_df = self.apply_features(df=df,features=features)
        ## calc rmses
        rmse_adj, rmse_base, rmse_static = self.calc_rmse(applied_df)
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