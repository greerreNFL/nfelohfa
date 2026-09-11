import pandas as pd


def define_surfaces(games, team_season, hfa_meta):
    '''
    Determines the field surface each team plays on and adds
    to the game df
    '''
    ## copy the games ##
    temp = games.copy()
    ## remove neutrals and unplayed games ##
    temp = temp[
        (temp['location'] != 'Neutral') &
        (~pd.isnull(temp['home_score']))
    ].copy()
    ## standardize turf types between artificial and natural ##
    temp['surface'] = temp['surface'].replace(hfa_meta['surface_repl'])
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
        team_season.copy(),
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


def add_surfaces(games, surfaces):
    '''
    add surface info to games
    Joins home_local_surface and away_local_surface.
    The dif_surface on/off flag is built by the feature.
    '''
    ## home ##
    games = pd.merge(
        games,
        surfaces.rename(columns={
            'team' : 'home_team',
            'surface' : 'home_local_surface'
        }),
        on=['home_team', 'season'],
        how='left'
    )
    ## away ##
    games = pd.merge(
        games,
        surfaces.rename(columns={
            'team' : 'away_team',
            'surface' : 'away_local_surface'
        }),
        on=['away_team', 'season'],
        how='left'
    )
    return games
