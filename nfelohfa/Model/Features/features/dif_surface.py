from ..feature import Feature
from ....Data.utilities.meta import load_meta

import numpy


class DifSurface(Feature):
    name = 'dif_surface'

    def build(self, games):
        '''
        1 when the away team is unused to this surface and the
        game surface is the home team's usual surface.
        '''
        repl = load_meta()['surface_repl']
        return numpy.where(
            (
                games['home_local_surface'] !=
                games['away_local_surface']
            ) &
            (
                games['surface'].replace(repl) ==
                games['home_local_surface']
            ),
            1,
            0
        )
