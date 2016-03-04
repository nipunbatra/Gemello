import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from copy import deepcopy

st = pd.HDFStore(os.path.expanduser("~/wiki-all.h5"))

metadata_df = pd.read_csv("../data/input/dataport-metadata.csv",index_col=0)