import pandas as pd


def define_local_timezones(games, team_season, hfa_meta):
    '''
    defines the local timezone
    '''
    ## build a df of TZs ##
    tzs_struc = []
    for team, tz in hfa_meta['timezones'].items():
        ## add a tz for the current season, which will backfill
        ## all previous seasons with the same value, unless
        ## it hits an override
        tzs_struc.append({
            'team' : team,
            'season' : games['season'].max(),
            'local_tz' : tz
        })
    ## add the overrides ##
    for override in hfa_meta['timezone_overrides']:
        tzs_struc.append(override)
    ## add to a team season df ##
    df = pd.merge(
        team_season.copy(),
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


def add_tzs(games, tz):
    '''
    add local timezone info to games
    '''
    games = pd.merge(
        games,
        tz.rename(columns={
            'team' : 'home_team',
            'local_tz' : 'home_local_tz'
        }),
        on=['home_team', 'season'],
        how='left'
    )
    games = pd.merge(
        games,
        tz.rename(columns={
            'team' : 'away_team',
            'local_tz' : 'away_local_tz'
        }),
        on=['away_team', 'season'],
        how='left'
    )
    return games
