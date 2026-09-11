from ..feature import Feature

import numpy


class AwayBye(Feature):
    name = 'away_bye'

    def build(self, games):
        ## determine byes ##
        return numpy.where(
            games['week'] > games['away_previous_week'] + 1,
            1,
            0
        )
