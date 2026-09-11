from ..feature import Feature
from ....Data.utilities.meta import load_meta

import pandas as pd
import numpy


class HomeTimeAdvantage(Feature):
    name = 'home_time_advantage'

    def build(self, games, peak_time='14:00'):
        '''
        Adds the net difference between each team
        and their local circadian optimal
        Intermediate cols (home_optimal_in_et, away_optimal_in_et,
        gametimestamp) are left on the frame, same as the old loader.
        '''
        tz_deltas = load_meta()['tz_deltas']
        ## home ##
        games['home_optimal_in_et'] = (
            pd.Timestamp(peak_time) +
            pd.Series([
                pd.Timedelta(hours=offset) for offset in
                games['home_local_tz'].map(tz_deltas)
            ])
        ).dt.time
        ## away ##
        games['away_optimal_in_et'] = (
            pd.Timestamp(peak_time) +
            pd.Series([
                pd.Timedelta(hours=offset) for offset in
                games['away_local_tz'].map(tz_deltas)
            ])
        ).dt.time
        ## kickoff ##
        games['gametimestamp'] = pd.to_datetime(
            games['gametime'],
            format = '%H:%M'
        ).dt.time
        ## define advantage ##
        return numpy.round(
            ## away dif from optimal in hours ##
            numpy.absolute(
                (
                    pd.to_datetime(games['gametimestamp'], format='%H:%M:%S') -
                    pd.to_datetime(games['away_optimal_in_et'], format='%H:%M:%S')
                ) / numpy.timedelta64(1, 'h')
            ) -
            ## less home dif from optimal in hours ##
            numpy.absolute(
                (
                    pd.to_datetime(games['gametimestamp'], format='%H:%M:%S') -
                    pd.to_datetime(games['home_optimal_in_et'], format='%H:%M:%S')
                ) / numpy.timedelta64(1, 'h')
            )
        ).fillna(0)
