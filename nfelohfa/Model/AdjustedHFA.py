import numpy

from .Features import FEATURES


class AdjustedHFA:
    '''
    Class for applying features that adjust BaseHFA.
    Training lives on Optimizer.
    '''
    def __init__(self, games, features):
        self.games = games.copy()
        self.features = features

    def apply_features(self, df=None, features=None):
        '''
        Applies features to the base hfa value
        '''
        ## if no features are passed, use the ones passed on init
        ## this is done so other features can be passed
        ## to the func with teh optimizer
        if features is None:
            features=self.features
        if df is None:
            df=self.games
        ## init hfa_adj ##
        df['hfa_adj'] = numpy.round(df['hfa_base'],3)
        for k,v in features.items():
            ## calc adj
            feat = FEATURES[k](weight=v)
            df['{0}_adj'.format(k)] = feat.apply(df['hfa_base'], df[k])
            ## add to hfa_adj ##
            df['hfa_adj'] = numpy.round(df['hfa_adj'] + df['{0}_adj'.format(k)],3)
        ## if the field is neutral, remove the base, leaving only the adjs, which we
        ## hypothesize to still be relevant ##
        df['hfa_adj'] = numpy.where(
            df['location'] == 'Neutral',
            df['hfa_adj'] - df['hfa_base'],
            df['hfa_adj']
        )
        ## then also zero out the base for clarity ##
        df['hfa_base'] = numpy.where(
            df['location'] == 'Neutral',
            0,
            df['hfa_base']
        )
        ## round hfa_adj to 3 decimal places ##
        df['hfa_adj'] = df['hfa_adj'].round(3)
        ## return ##
        return df
