import json
import pathlib

PACKAGE_DIR = pathlib.Path(__file__).parent.parent.resolve()


class Config:
    '''
    Package config loaded from parameters.json.
    team_strength: qbelo_base and elo_per_point
    base: rolling HFA params
    features: name -> weight
    '''
    def __init__(self, team_strength, base, features):
        self.team_strength = team_strength
        self.base = base
        self.features = features

    @classmethod
    def from_dict(cls, data):
        return cls(
            team_strength=data['team_strength'],
            base=data['base'],
            features=data['features']
        )

    def to_dict(self):
        return {
            'team_strength': self.team_strength,
            'base': self.base,
            'features': self.features
        }

    @classmethod
    def load(cls, path=None):
        if path is None:
            path = '{0}/parameters.json'.format(PACKAGE_DIR)
        with open(path) as fp:
            return cls.from_dict(json.load(fp))

    def save(self, path=None):
        if path is None:
            path = '{0}/parameters.json'.format(PACKAGE_DIR)
        with open(path, 'w') as fp:
            json.dump(self.to_dict(), fp, indent=4)
