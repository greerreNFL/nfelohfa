import numpy


class Feature:
    '''
    One HFA feature: build the column from games, apply a weight
    to hfa_base. Spec (weight, apply mode) lives on the instance.
    '''
    name = None
    apply_mode = 'multiply'

    def __init__(self, weight=0):
        self.weight = weight

    def build(self, games):
        '''
        Return a series aligned to games: the feature value
        (flag or continuous).
        '''
        raise NotImplementedError

    def apply(self, hfa_base, values):
        '''
        Turn the feature column into an HFA adjustment.
        multiply: hfa_base * values * weight
        add: values * weight
        '''
        if self.apply_mode == 'multiply':
            ## if not active, values will be 0
            return numpy.round((
                hfa_base * (values * self.weight)
            ), 3)
        if self.apply_mode == 'add':
            return numpy.round((
                values * self.weight
            ), 3)
        raise ValueError(
            'Unknown apply_mode {0} for {1}'.format(
                self.apply_mode, self.name
            )
        )
