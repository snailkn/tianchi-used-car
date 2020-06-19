import numpy as np
import pandas as pd


def date_parser(x):
    try:
        return pd.datetime.strptime(x, "%Y%m%d")
    except:
        return np.nan
