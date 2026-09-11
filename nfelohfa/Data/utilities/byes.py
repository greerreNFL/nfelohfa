import pandas as pd


def define_previous_weeks(games):
    '''
    Builds each team's previous game week from the schedule.
    One row per team-game. Shared by home and away bye joins.
    '''
    ## flatten ##
    flat = pd.concat([
        games[[
            'season', 'week', 'home_team'
        ]].rename(columns={'home_team':'team'}),
        games[[
            'season', 'week', 'away_team'
        ]].rename(columns={'away_team':'team'})
    ]).sort_values(
        by=['season', 'team', 'week']
    ).reset_index(drop=True)
    ## get week of previous game ##
    flat['previous_week'] = flat.groupby(['team', 'season'])['week'].shift(1)
    return flat[['season', 'week', 'team', 'previous_week']]


def add_previous_weeks(games, previous_weeks):
    '''
    add previous week info to games
    Joins home_previous_week and away_previous_week.
    The bye on/off flag is built by the feature.
    '''
    ## add to games ##
    ## home ##
    games = pd.merge(
        games,
        previous_weeks.rename(columns={
            'team' : 'home_team',
            'previous_week' : 'home_previous_week'
        }),
        on=['season', 'week', 'home_team'],
        how='left'
    )
    ## away ##
    games = pd.merge(
        games,
        previous_weeks.rename(columns={
            'team' : 'away_team',
            'previous_week' : 'away_previous_week'
        }),
        on=['season', 'week', 'away_team'],
        how='left'
    )
    return games
