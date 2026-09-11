from ..feature import Feature

import numpy


class HomeBye(Feature):
    name = 'home_bye'

    def build(self, games):
        ## determine byes ##
        return numpy.where(
            games['week'] > games['home_previous_week'] + 1,
            1,
            0
        )
