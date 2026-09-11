import pandas as pd
import numpy
import pathlib

import nfelodcm as dcm

from .utilities import (
    load_meta, define_surfaces, add_surfaces,
    define_local_timezones, add_tzs,
    define_weekly_ratings, add_weekly_ratings,
    define_previous_weeks, add_previous_weeks
)

class DataLoader():
    '''
    Loads and stores all data
      * Retrieves game file
      * Adds tz, temps, travel dist, etc
    Feature on/off flags (bye, surface mismatch, time advantage)
    are built by Model.Features, not here.
    '''
    def __init__(self):
        self.db = dcm.load(['games', 'srs_ratings']) ## will also want to load power ratings ##
        ## load local data ##
        self.package_loc = pathlib.Path(__file__).parent.parent.parent.resolve()
        self.data_dir = pathlib.Path(__file__).parent.resolve()
        self.hfa_meta = load_meta()
        self.weekly_temps = self.load_temps()
        ## create stuctures ##
        self.team_season = self.build_team_season()
        self.surfaces = define_surfaces(
            self.db['games'], self.team_season, self.hfa_meta
        )
        self.tz = define_local_timezones(
            self.db['games'], self.team_season, self.hfa_meta
        )
        self.team_ratings = define_weekly_ratings(self.db['srs_ratings'])
        self.previous_weeks = define_previous_weeks(self.db['games'])
        ## add data to games ##
        self.db['games'] = add_surfaces(self.db['games'], self.surfaces)
        self.db['games'] = add_tzs(self.db['games'], self.tz)
        self.db['games'] = add_weekly_ratings(
            self.db['games'], self.team_ratings
        )
        self.add_local_temps()
        self.db['games'] = add_previous_weeks(
            self.db['games'], self.previous_weeks
        )
        self.add_div()

    def load_temps(self):
        '''
        loads weekly temperatures for each team location
        '''
        return pd.read_csv(
            '{0}/temps_by_week.csv'.format(self.data_dir),
            index_col=0
        )

    def build_team_season(self):
        '''
        Builds a structure for each unique team<>season combo
        '''
        all_team_struc = []
        for season in range(
            self.db['games']['season'].min(),
            self.db['games']['season'].max() + 1
        ):
            for team in self.db['games']['home_team'].unique():
                all_team_struc.append({
                    'team' : team,
                    'season' : season
                })
        ## create df ##
        df = pd.DataFrame(all_team_struc)
        return df.sort_values(
            by=['team', 'season'],
            ascending=[True, True]
        ).reset_index(drop=True)

    def add_local_temps(self):
        '''
        defines local temp by week
        '''
        ## helper for handling location changes in a vectorized ##
        ## way ##
        def get_weather(team_col):
            ## make a copy of games
            temp = self.db['games'].copy()
            ## change team name ##
            for override in self.hfa_meta['weather_location_overrides']:
                if override['direction'] == 'gt':
                    temp[team_col] = numpy.where(
                        (temp[team_col] == override['team']) &
                        (temp['season'] > override['season']),
                        override['repl'],
                        temp[team_col]
                    )
                else:
                    temp[team_col] = numpy.where(
                        (temp[team_col] == override['team']) &
                        (temp['season'] < override['season']),
                        override['repl'],
                        temp[team_col]
                    )
            ## join the weather ##
            temp = pd.merge(
                temp,
                self.weekly_temps.rename(columns={
                    'team' : team_col
                }),
                on=['week', team_col],
                how='left'
            )
            ## return ##
            return temp['week_temp']
        ## add to games ##
        self.db['games']['home_local_temp'] = get_weather('home_team')
        self.db['games']['away_local_temp'] = get_weather('away_team')
        self.db['games']['absolute_temperature_difference'] = numpy.absolute(
            self.db['games']['home_local_temp'] -
            self.db['games']['away_local_temp']
        )

    def add_div(self):
        '''
        Adds boolean for div and non div fields
        '''
        self.db['games']['non_div_game'] = numpy.where(
            self.db['games']['div_game'] == 0,
            1,
            0
        )
