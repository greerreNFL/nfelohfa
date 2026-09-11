from .features.home_bye import HomeBye
from .features.away_bye import AwayBye
from .features.home_time_advantage import HomeTimeAdvantage
from .features.dif_surface import DifSurface
from .features.div_game import DivGame

FEATURES = {
    'home_bye' : HomeBye,
    'away_bye' : AwayBye,
    'home_time_advantage' : HomeTimeAdvantage,
    'dif_surface' : DifSurface,
    'div_game' : DivGame,
}


def build_all(games):
    '''
    Add every registered feature column to games.
    '''
    for name, cls in FEATURES.items():
        games[name] = cls(weight=0).build(games)
    return games
