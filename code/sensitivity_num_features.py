# NEED TO RUN ON CLUSTER
import sys
CLUSTER=True
if CLUSTER:
    sys.path.insert(0, '/if6/nb2cz/anaconda/lib/python2.7/site-packages')



from create_df import read_df
df, dfc, all_homes, appliance_min, national_average = read_df()

from all_functions import *
from features import *





import sys
appliance, feature = sys.argv[1], sys.argv[2]
out_mean, optimal_dict_mean = find_optimal_features(df, dfc, all_homes, appliance_min, national_average,
                                                        [appliance],
                                                        {feature:feature_map[feature]},
                          NUM_NEIGHBOURS_MAX=7, F_length_max=7, metric="mean",
                          train_outlier=True, test_outlier=False, outlier_features=None, outlier_fraction=0.1,
                            print_steps=False)

import json
json.dump(optimal_dict_mean, open("%s_%s" %(appliance, feature), 'r'))

