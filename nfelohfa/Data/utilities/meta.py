import json
import pathlib

DATA_DIR = pathlib.Path(__file__).parent.parent.resolve()


def load_meta():
    '''
    loads the meta json
    '''
    with open('{0}/meta.json'.format(DATA_DIR), 'r') as fp:
        return json.load(fp)
