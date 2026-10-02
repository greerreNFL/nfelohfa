from ..feature import Feature

import numpy


class Week1(Feature):
    name = 'week_1'

    def build(self, games):
        return numpy.where(
            games['week'] == 1,
            1,
            0
        )
