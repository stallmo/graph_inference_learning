import os

import pandas as pd

def load():
    """
    Loads (cleaned) german parliament voting data.

    :return: pandas DataFrame.
    """
    dirname = os.path.dirname(__file__)

    return pd.read_csv(os.path.join(dirname, 'parliament_voting.csv'))