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
            return hfa_base * (values * self.weight)
        if self.apply_mode == 'add':
            return values * self.weight
        raise ValueError(
            'Unknown apply_mode {0} for {1}'.format(
                self.apply_mode, self.name
            )
        )
