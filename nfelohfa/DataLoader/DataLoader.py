import pandas as pd 
import numpy
import pathlib
import json

import nfelodcm as dcm

class DataLoader():
    '''
    Loads and stores all data
      * Retrieves game file
      * Adds tz, temps, travel dist, etc
    '''
    def __init__(self):
        self.db = dcm.load(['games', 'srs_ratings']) ## will also want to load power ratings ##
        ## load local data ##
        self.package_loc = pathlib.Path(__file__).parent.parent.parent.resolve()
        self.hfa_meta = self.load_meta()
        self.weekly_temps = self.load_temps()
        ## create stuctures ##
        self.team_season = self.build_team_season()
        self.surfaces = self.define_surfaces()
        self.tz = self.define_local_timezones()
        self.team_ratings = self.define_weekly_ratings()
        ## add data to games ##
        self.add_surfaces()
        self.add_tzs()
        self.add_weekly_ratings()
        self.add_home_time_advantage()
        self.add_local_temps()
        self.add_byes()
        self.add_div()
    
    def load_meta(self):
        '''
        loads the meta json
        '''
        with open(
            '{0}/nfelohfa/DataLoader/meta.json'.format(
                self.package_loc
            ),
            'r'
        ) as fp:
            return json.load(fp)
    
    def load_temps(self):
        '''
        loads weekly temperatures for each team location
        '''
        return pd.read_csv(
            '{0}/nfelohfa/DataLoader/temps_by_week.csv'.format(
                self.package_loc
            ),
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
    
    def define_surfaces(self):
        '''
        Determines the field surface each team plays on and adds
        to the game df
        '''
        ## copy the games ##
        temp = self.db['games'].copy()
        ## remove neutrals and unplayed games ##
        temp = temp[
            (temp['location'] != 'Neutral') &
            (~pd.isnull(temp['home_score']))
        ].copy()
        ## standardize turf types between artificial and natural ##
        temp['surface'] = temp['surface'].replace(self.hfa_meta['surface_repl'])
        ## generate a df of fieldtypes by team and season ##
        fields = temp.groupby(
            ['home_team', 'season', 'surface']
        ).agg(
            games_played = ('home_score', 'count')
        ).reset_index()
        ## get the most played surface ##
        fields = fields.sort_values(
            by=['games_played'],
            ascending=[False]
        ).reset_index(drop=True).groupby(
            ['home_team', 'season']
        ).head(1)
        ## add fields to a team<>seaosn df ##
        df = pd.merge(
            self.team_season.copy(),
            fields[[
                'home_team', 'season', 'surface'
            ]].rename(columns={
                'home_team' : 'team'
            }),
            on=['team', 'season'],
            how='left'
        )
        ## fill missing ##
        df = df.sort_values(
            by=['team', 'season'],
            ascending=[True, True]
        ).reset_index(drop=True)
        df['surface'] = df.groupby(
            ['team']
        )['surface'].transform(lambda x: x.bfill().ffill())
        ## return ##
        return df

    def define_local_timezones(self):
        '''
        defines the local timezone
        '''
        ## build a df of TZs ##
        tzs_struc = []
        for team, tz in self.hfa_meta['timezones'].items():
            ## add a tz for the current season, which will backfill
            ## all previous seasons with the same value, unless
            ## it hits an override
            tzs_struc.append({
                'team' : team,
                'season' : self.db['games']['season'].max(),
                'local_tz' : tz
            })
        ## add the overrides ##
        for override in self.hfa_meta['timezone_overrides']:
            tzs_struc.append(override)
        ## add to a team season df ##
        df = pd.merge(
            self.team_season.copy(),
            pd.DataFrame(tzs_struc).groupby(
                ['team', 'season']
            ).head(1),
            on=['team', 'season'],
            how='left'
        )
        ## backfill ##
        df['local_tz'] = df.groupby(
            ['team']
        )['local_tz'].transform(lambda x: x.bfill())
        ## return ##
        return df 

    def define_weekly_ratings(self):
        '''
        Adds weekly team ratings from the SRS pacakge, which are used
        to create an opponent adjusted expectation
        '''
        ## need to shift ratings forward 1 week since they are through
        ## the end of the week, while preserving the QB adj, which is
        ## of the correct week
        self.db['srs_ratings']['proj_rating'] = self.db['srs_ratings'].groupby([
            'team', 'season'
        ])['srs_rating_normalized'].shift(1)
        ## add the preseason wt rating for the first week, which is nan due to
        ## the shift ##
        self.db['srs_ratings']['proj_rating'] = self.db['srs_ratings']['proj_rating'].combine_first(
            self.db['srs_ratings']['pre_season_wt_rating']
        )
        ## add the qb adj ##
        self.db['srs_ratings']['proj_rating'] = (
            self.db['srs_ratings']['proj_rating'] + 
            self.db['srs_ratings']['qb_adjustment']
        )
        ## return ##
        return self.db['srs_ratings'][[
            'season', 'week', 'team', 'proj_rating'
        ]]

    def add_surfaces(self):
        '''
        add surface info to games
        '''
        ## home ##
        self.db['games'] = pd.merge(
            self.db['games'],
            self.surfaces.rename(columns={
                'team' : 'home_team',
                'surface' : 'home_local_surface'
            }),
            on=['home_team', 'season'],
            how='left'
        )
        ## away ##
        self.db['games'] = pd.merge(
            self.db['games'],
            self.surfaces.rename(columns={
                'team' : 'away_team',
                'surface' : 'away_local_surface'
            }),
            on=['away_team', 'season'],
            how='left'
        )
        self.db['games']['dif_surface'] = numpy.where(
            (
                self.db['games']['home_local_surface'] !=
                self.db['games']['away_local_surface']
            ) &
            (
                self.db['games']['surface'].replace(self.hfa_meta['surface_repl']) ==
                self.db['games']['home_local_surface']
            ),
            1,
            0
        )
    
    def add_tzs(self):
        '''
        add local timezone info to games
        '''
        self.db['games'] = pd.merge(
            self.db['games'],
            self.tz.rename(columns={
                'team' : 'home_team',
                'local_tz' : 'home_local_tz'
            }),
            on=['home_team', 'season'],
            how='left'
        )
        self.db['games'] = pd.merge(
            self.db['games'],
            self.tz.rename(columns={
                'team' : 'away_team',
                'local_tz' : 'away_local_tz'
            }),
            on=['away_team', 'season'],
            how='left'
        )
    
    def add_weekly_ratings(self):
        '''
        Adds the weekly ratings to the file
        '''
        self.db['games'] = pd.merge(
            self.db['games'],
            self.team_ratings.rename(columns={
                'team' : 'home_team',
                'proj_rating' : 'home_team_rating'
            }),
            on=['season', 'week', 'home_team'],
            how='left'
        )
        self.db['games'] = pd.merge(
            self.db['games'],
            self.team_ratings.rename(columns={
                'team' : 'away_team',
                'proj_rating' : 'away_team_rating'
            }),
            on=['season', 'week', 'away_team'],
            how='left'
        )
    
    def add_home_time_advantage(self, peak_time='14:00'):
        '''
        Adds the net difference between each team
        and their local circadian optimal
        '''
        ## home ##
        self.db['games']['home_optimal_in_et'] = (
            pd.Timestamp(peak_time) +
            pd.Series([
                pd.Timedelta(hours=offset) for offset in
                self.db['games']['home_local_tz'].map(
                    self.hfa_meta['tz_deltas']
                )
            ])
        ).dt.time
        ## away ##
        self.db['games']['away_optimal_in_et'] = (
            pd.Timestamp(peak_time) +
            pd.Series([
                pd.Timedelta(hours=offset) for offset in
                self.db['games']['away_local_tz'].map(
                    self.hfa_meta['tz_deltas']
                )
            ])
        ).dt.time
        ## kickoff ##
        self.db['games']['gametimestamp'] = pd.to_datetime(
            self.db['games']['gametime'],
            format = '%H:%M'
        ).dt.time
        ## define advantage ##
        self.db['games']['home_time_advantage'] = numpy.round(
            ## away dif from optimal in hours ##
            numpy.absolute(
                (
                    pd.to_datetime(self.db['games']['gametimestamp'], format='%H:%M:%S') -
                    pd.to_datetime(self.db['games']['away_optimal_in_et'], format='%H:%M:%S')
                ) / numpy.timedelta64(1, 'h')
            ) -
            ## less home dif from optimal in hours ##
            numpy.absolute(
                (
                    pd.to_datetime(self.db['games']['gametimestamp'], format='%H:%M:%S') -
                    pd.to_datetime(self.db['games']['home_optimal_in_et'], format='%H:%M:%S')
                ) / numpy.timedelta64(1, 'h')
            )
        ).fillna(0)

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
    
    def add_byes(self):
        '''
        Adds bye weeks for how and away
        '''
        ## flatten ##
        flat = pd.concat([
            self.db['games'][[
                'season', 'week', 'home_team'
            ]].rename(columns={'home_team':'team'}),
            self.db['games'][[
                'season', 'week', 'away_team'
            ]].rename(columns={'away_team':'team'})
        ]).sort_values(
            by=['season', 'team', 'week']
        ).reset_index(drop=True)
        ## get week of previous game ##
        flat['previous_week'] = flat.groupby(['team', 'season'])['week'].shift(1)
        ## determine byes ##
        flat['bye_week']=numpy.where(
            flat['week'] > flat['previous_week'] + 1,
            1,
            0
        )
        ## add to games ##
        self.db['games'] = pd.merge(
            self.db['games'],
            flat[['season', 'week', 'team', 'bye_week']].rename(columns={
                'team' : 'home_team',
                'bye_week' : 'home_bye'
            }),
            on=['season', 'week', 'home_team'],
            how='left'
        )
        self.db['games'] = pd.merge(
            self.db['games'],
            flat[['season', 'week', 'team', 'bye_week']].rename(columns={
                'team' : 'away_team',
                'bye_week' : 'away_bye'
            }),
            on=['season', 'week', 'away_team'],
            how='left'
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