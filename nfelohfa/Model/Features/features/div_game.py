from ..feature import Feature


class DivGame(Feature):
    name = 'div_game'

    def build(self, games):
        return games['div_game']
