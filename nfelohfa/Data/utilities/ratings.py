import pandas as pd


def define_weekly_ratings(qbelo, team_strength):
    '''
    Weekly team ratings from qbelo, in points.

    team_strength is the config section. For a regular-season week t,
    team Elo is the average of elo_pre from t through
    t + forward_window. The end is clipped at the last regular-season
    week in that season. Bye weeks in the span are omitted. The week-t
    quarterback piece, qbelo_pre - elo_pre, is added after the average.
    Playoff rows keep the week-t qbelo rating.
    '''
    forward_window = team_strength['forward_window']
    qbelo_base = team_strength['qbelo_base']
    elo_per_point = team_strength['elo_per_point']
    qbelo = qbelo[
        qbelo['qbelo1_pre'].notna() &
        qbelo['elo1_pre'].notna() &
        qbelo['week'].notna()
    ].copy()
    qbelo = qbelo.sort_values(
        by=['season', 'week', 'date']
    ).reset_index(drop=True)
    ## mid-week file can carry two rows for one game_id ##
    qbelo = qbelo[
        qbelo['game_id'].isna() | ~qbelo['game_id'].duplicated(keep='last')
    ].copy()
    home = qbelo[[
        'season', 'week', 'game_type', 'team1', 'elo1_pre', 'qbelo1_pre'
    ]].rename(columns={
        'team1' : 'team',
        'elo1_pre' : 'elo_pre',
        'qbelo1_pre' : 'qbelo_pre'
    })
    away = qbelo[[
        'season', 'week', 'game_type', 'team2', 'elo2_pre', 'qbelo2_pre'
    ]].rename(columns={
        'team2' : 'team',
        'elo2_pre' : 'elo_pre',
        'qbelo2_pre' : 'qbelo_pre'
    })
    ratings = pd.concat([home, away], ignore_index=True)
    ratings['week'] = ratings['week'].astype(int)
    ## one row per team-week. Earlier game in the week wins ##
    ratings = ratings.groupby(
        ['season', 'week', 'team']
    ).head(1).reset_index(drop=True)
    reg = ratings[ratings['game_type'] == 'REG'].copy()
    other = ratings[ratings['game_type'] != 'REG'].copy()
    later = reg[[
        'season', 'team', 'week', 'elo_pre'
    ]].rename(columns={
        'week' : 'later_week',
        'elo_pre' : 'later_elo'
    })
    span = pd.merge(reg, later, on=['season', 'team'], how='left')
    span = span[
        (span['later_week'] >= span['week']) &
        (span['later_week'] <= span['week'] + forward_window)
    ]
    elo_mean = span.groupby(
        ['season', 'week', 'team']
    )['later_elo'].mean().reset_index()
    reg = pd.merge(reg, elo_mean, on=['season', 'week', 'team'], how='left')
    reg['proj_rating'] = (
        (reg['later_elo'] - qbelo_base) +
        (reg['qbelo_pre'] - reg['elo_pre'])
    ) / elo_per_point
    other['proj_rating'] = (
        other['qbelo_pre'] - qbelo_base
    ) / elo_per_point
    out = pd.concat([reg, other], ignore_index=True)
    return out[[
        'season', 'week', 'team', 'proj_rating'
    ]].reset_index(drop=True)


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
