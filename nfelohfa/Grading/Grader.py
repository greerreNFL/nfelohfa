class Grader:
    '''
    RMSE of opponent-adjusted margin vs adj HFA, rolling base, and
    a static HFA level.
    '''
    def __init__(self, level=2.5):
        self.level = level

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
