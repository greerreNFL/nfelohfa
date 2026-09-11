import json
import pathlib

PACKAGE_DIR = pathlib.Path(__file__).parent.parent.resolve()


class Config:
    '''
    Package config loaded from parameters.json.
    base: rolling HFA params
    features: name -> weight
    '''
    def __init__(self, base, features):
        self.base = base
        self.features = features

    @classmethod
    def from_dict(cls, data):
        return cls(
            base=data['base'],
            features=data['features']
        )

    def to_dict(self):
        return {
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
