import pandas as pd
import numpy

import statsmodels.api as sm
from scipy.optimize import minimize
import pathlib

class BaseHFA:
    '''
    Calculates the baseline HFA by week overtime with no
    additional adjustments.

    The model is similar to a LOESS. For each week, it calculates
    a rolling regression based on a backwards looking window.

    This expectation for HFA is then passed to an EMA to calculate a
    smoothed expectatin for HFA that is less susceptible to noisy results

    It takes:
    * games, which is loaded by the DataLoader
    * level_weeks, which is the span term in the EMA
    * reg_weeks, which is the number of weeks to include in the regression
    * kick_in, which determines what % of the reg_weeks are required before
    ** the model starts updating the rolling HFA

    Includes a class for training the parameters as well using brute force loop
    Over ranges for each variable
    '''
    def __init__(self,
        games, level_weeks, reg_weeks,
        level=2.5, kick_in=.75, optimize=False
    ):
        self.folder_loc = pathlib.Path(__file__).parent.resolve()
        self.all_weeks = games[[
            'season', 'week'
        ]].drop_duplicates().reset_index(drop=True)
        self.games = games.copy()
        self.filtered_games = self.prep_games(games)
        self.forward_hfa = self.calc_forward_hfa()
        ## params ##
        self.level_weeks = level_weeks
        self.reg_weeks = reg_weeks
        self.level = level
        self.kick_in = kick_in
        ## outputs ##
        self.hfa_df = self.calc_hfa() if not optimize else None
        self.games_w_hfa = self.add_hfa() if not optimize else None
        self.optimal_params = self.optimize() if optimize else None
        self.optimizer_results = None
        
    def prep_games(self, games):
        '''
        Filters unplayed games, playoffs, and games with no ratings
        Filters covid so as not to scew results
        '''
        temp = games.copy()
        ## create a simple expected result from ratings ##
        temp['expected_result'] = (
            temp['home_team_rating'] -
            temp['away_team_rating']
        )
        temp['home_margin_error'] = (
            temp['result'] -
            temp['expected_result']
        )
        return temp[
            (~pd.isnull(temp['expected_result'])) &
            (temp['season'] != 2020) &
            (temp['game_type'] == 'REG') &
            (temp['location'] != 'Neutral')
        ].copy().reset_index(drop=True)

    def calc_forward_hfa(self, window=20):
        '''
        Calculates a forward looking HFA to use in optimization
        '''
        f_hfa = self.filtered_games.groupby(['season', 'week']).agg(
            avg_home_error = ('home_margin_error', 'mean')
        ).reset_index()
        ## forward window ##
        f_hfa['forward_observed_hfa'] = f_hfa['avg_home_error'].rolling(window).mean().shift(-window)
        f_hfa['trailing_observed_hfa'] = f_hfa['avg_home_error'].rolling(window).mean()
        return f_hfa

    def calc_hfa_sub(self, level, level_weeks, reg_weeks, kick_in=0.5):
        '''
        Main function for calculating the rolling HFA
        This takes args vs the class properties so it can
        be used by the optimizer as well

        To calcualte HFA with class properties, use the calc_hfa
        wrapper
        '''
        temp = self.filtered_games.groupby(['season', 'week']).agg(
            avg_error = ('home_margin_error', 'mean')
        ).reset_index()
        ## init params ##
        temp['level'] = level
        a = 2 / (level_weeks + 1)
        temp['intercept_constant'] = 1
        ## regression ##
        ## goes week by week and calcs regressions
        for index, row in temp.iterrows():
            ## cant calc if window and ema does not have enough n
            ## since the best regression is likely to be very long,
            ## the model wont "kick in" until well into the data set,
            ## even tho a downward trend will exist before then
            ## to compensate, allow level to update, but scale down
            ## alpha based on how close we are to the reg_week variable
            if index < kick_in * reg_weeks or index < level_weeks:
                pass
            else:
                ## since this will update before reg_weeks is reached,
                ## create discounted vars ##
                window_start = int(max(index-reg_weeks, 0))
                a_ = a * min(index/reg_weeks, 1)
                ## window data ##
                trailing_window = temp.iloc[
                    window_start:index
                ].copy()
                trailing_window['week_num'] = numpy.arange(
                    len(trailing_window)
                ) + 1
                ## fit ##
                reg = sm.OLS(
                    trailing_window['avg_error'],
                    trailing_window[['week_num','intercept_constant']],
                    hasconst=True
                ).fit()
                ## get the expected value for the most recent week ##
                update_val = (
                    reg.params.intercept_constant +
                    (
                        trailing_window['week_num'].max() *
                        reg.params.week_num
                    )
                )
                ## Update the EMA for this value ##
                ## get previous value ##
                prev_level = temp.iloc[index-1]['level']
                ## update prev week value ##
                temp.loc[index, 'level'] = (
                    a_ * update_val +
                    (1-a_) * prev_level
                )
        ## level is through the end of the week, so shift forward ##
        temp['rolling_hfa'] = temp['level'].shift(1).fillna(self.level)
        ## return ##
        return temp

    def calc_hfa(self):
        '''
        Calcualte HFA and return a DF of season, week, hfa_base
        '''
        ## get the HFA calc ##
        proj_hfa = self.calc_hfa_sub(
            self.level, self.level_weeks,
            self.reg_weeks, self.kick_in
        )
        ## add to season weeks ##
        hfa = pd.merge(
            self.all_weeks,
            proj_hfa.rename(columns={
                'rolling_hfa' : 'hfa_base'
            }),
            on=['season', 'week'],
            how='left'
        )
        ## Fill all the missing vals ##
        ## 2020 ##
        hfa['hfa_base'] = numpy.where(
            hfa['season'] == 2020,
            0.25,
            hfa['hfa_base']
        )
        ## ffill for playoffs ##
        hfa['hfa_base'] = hfa['hfa_base'].ffill()
        ## fill rest with baseline level ##
        hfa['hfa_base'] = hfa['hfa_base'].fillna(self.level)
        ## return ##
        return hfa

    def add_hfa(self):
        '''
        Add HFA to games
        '''
        hfa = pd.merge(
            self.games,
            self.hfa_df[[
                'season', 'week', 'hfa_base'
            ]],
            on=['season', 'week'],
            how='left'
        )
        ## add forward and trailing ##
        hfa = pd.merge(
            hfa,
            self.forward_hfa,
            on=['season', 'week'],
            how='left',
        )
        ## return ##
        return hfa
    
    
    def obj_func(self, x):
        '''
        Objective function for the optimizers using MAE
        for opponent + hfa results against actual
        '''
        ## add hfa to games ##
        temp = pd.merge(
            self.games,
            self.calc_hfa_sub(
                x[0],
                x[1],
                x[2],
                x[3]
            ),
            on=['season', 'week'],
            how='left'
        )
        ## drop missing hfas ##
        temp = temp[~pd.isnull(temp['rolling_hfa'])].copy()
        ## calc error ##
        temp['absolute_error'] = numpy.absolute(
            ## expected result ##
            (
                temp['home_team_rating'] +
                temp['rolling_hfa'] -
                temp['away_team_rating']
            ) -
            ## less actual result ##
            temp['result']
        )
        ## return MAE ##
        return temp['absolute_error'].mean()

    def optimize(self):
        '''
        Brute force optimizer for level and reg_weeks
        '''
        recs = []
        for kick_in in [.5, .75, 1]:
            for level in [2.0, 2.25, 2.5, 2.75]:
                for level_week in range(10,200):
                    for reg_week in range(60,300):
                        recs.append({
                            'level' : level,
                            'level_weeks' : level_week,
                            'reg_weeks' : reg_week,
                            'kick_in' : kick_in,
                            'mae' : self.obj_func([
                                float(level), level_week,
                                reg_week, kick_in
                            ])
                        })
                        ## save every 10k records ##
                        if len(recs) % 10000 == 0:
                            print('Through {0} combinations. Saving...'.format(
                                len(recs)
                            ))
                            pd.DataFrame(recs).sort_values(
                                by=['mae'], ascending=[True]
                            ).reset_index(drop=True).to_csv(
                                '{0}/optimizer_results.csv'.format(
                                    self.folder_loc
                                )
                            )
        ## df ##
        df = pd.DataFrame(recs)
        df = df.sort_values(by=['mae'], ascending=[True]).reset_index(drop=True)
        self.optimizer_results=df
        self.optimizer_results.to_csv(
            '{0}/optimizer_results.csv'.format(
                self.folder_loc
            )
        )
        self.optimizer_results.head(20000).to_csv(
            '{0}/optimizer_results_top_20k.csv'.format(
                self.folder_loc
            )
        )
        ## create an hfa df using the best config ##
        self.level = df.iloc[0]['level']
        self.level_weeks = df.iloc[0]['level_weeks']
        self.reg_weeks = df.iloc[0]['reg_weeks']
        self.kick_in = df.iloc[0]['kick_in']
        best = self.calc_hfa()
        ## join best to windows ##
        best = pd.merge(
            self.forward_hfa,
            best,
            on=['season', 'week'],
            how='left'
        )
        best.to_csv(
            '{0}/optimized_hfa_base.csv'.format(
                self.folder_loc
            )
        )
        return {
            'level' : self.level,
            'level_weeks' : self.level_weeks,
            'reg_weeks' : self.reg_weeks,
            'kick_in' : self.kick_in,
        }
