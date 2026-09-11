import pandas as pd


def define_weekly_ratings(srs_ratings):
    '''
    Adds weekly team ratings from the SRS pacakge, which are used
    to create an opponent adjusted expectation
    '''
    ## need to shift ratings forward 1 week since they are through
    ## the end of the week, while preserving the QB adj, which is
    ## of the correct week
    srs_ratings['proj_rating'] = srs_ratings.groupby([
        'team', 'season'
    ])['srs_rating_normalized'].shift(1)
    ## add the preseason wt rating for the first week, which is nan due to
    ## the shift ##
    srs_ratings['proj_rating'] = srs_ratings['proj_rating'].combine_first(
        srs_ratings['pre_season_wt_rating']
    )
    ## add the qb adj ##
    srs_ratings['proj_rating'] = (
        srs_ratings['proj_rating'] +
        srs_ratings['qb_adjustment']
    )
    ## return ##
    ## there is an upstream duplicate in the srs ratings, so we need to
    ## group by season, week, and team and take the first
    return srs_ratings[[
        'season', 'week', 'team', 'proj_rating'
    ]].groupby(
        ['season', 'week', 'team']
    ).head(1).reset_index(drop=True)


def add_weekly_ratings(games, team_ratings):
    '''
    Adds the weekly ratings to the file
    '''
    games = pd.merge(
        games,
        team_ratings.rename(columns={
            'team' : 'home_team',
            'proj_rating' : 'home_team_rating'
        }),
        on=['season', 'week', 'home_team'],
        how='left'
    )
    games = pd.merge(
        games,
        team_ratings.rename(columns={
            'team' : 'away_team',
            'proj_rating' : 'away_team_rating'
        }),
        on=['season', 'week', 'away_team'],
        how='left'
    )
    return games
