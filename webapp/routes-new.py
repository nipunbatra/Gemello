from flask import Flask, render_template, json, request
import numpy as np

import sys
CLUSTER=True
if CLUSTER:
    sys.path.insert(0, '/if6/nb2cz/anaconda/lib/python2.7/site-packages')

import sys
sys.path.append("../code/")
import pandas as pd
import numpy as np

from create_df import read_df
df, dfc, all_homes, appliance_min, national_average = read_df()

from all_functions import *
from features import *

import json




from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

def predict_appliance(home, appliance, feature):
    if home in all_homes[appliance]:
        home_to_pick=home
    else:
        home_to_pick=all_homes[appliance][0]
    print home_to_pick


    #feature_dict = json.load(open("../data/output/sensitivity-numfeatures-allhomes/%s_%s_%d.json" %(appliance,feature, home_to_pick),"r"))
    #f = feature_dict['f']
    #k = feature_dict['k']
    f = ["num_rooms", "skew", "area", "aggregate_8", "ratio_difference_min_max", "aggregate_9"]
    k = 3
    clf = KNeighborsRegressor(n_neighbors=k)
    nn = NearestNeighbors(n_neighbors=k)
    df_new =df.copy()
    df_new = df_new.ix[all_homes[appliance]]
    df_new = df_new.ix[~df_new.index.isin([home])]
    #df_new = df_new.drop(home, axis=1)
    nn.fit(df_new[f].dropna())
    distances, indices = nn.kneighbors(df.ix[home][f])
    out = []
    nghbrs_list = df_new.index[indices].values[0]

    for month in range(1, 13):
        if len(nghbrs_list>1):
            out.append(df_new[["%s_%d" %(appliance, month) ]].ix[nghbrs_list].sum().values[0]/k)
        else:
            out.append(df_new[["%s_%d" %(appliance, month) ]].ix[nghbrs_list].values[0]/k)
    return out


all_appliances = ['fridge','hvac','wm','dr','dw','light']
global_df_aggregate = dfc[["aggregate_%d" %i for i in range(1, 13)]].sum(axis=1)
appliance_name_map = {'fridge':"Fridge",
                      'hvac':"HVAC",
                      'dr':'Dryer',
                      'wm':'Washing machine',
                      'dw':'Dish washer',
                      'light':'Light'
                      }
from scipy import stats

def predict(occupants, rooms, area, home_id):
    out = {}
    for appliance in all_appliances:
        out[appliance] = predict_appliance(home_id, appliance, "Monthly+Static")
    out['aggregate'] = dfc.ix[home_id][["aggregate_%d" %i for i in range(1, 13)]].squeeze().tolist()
    out_df = pd.DataFrame(out)
    out_df["Others"] = out_df["aggregate"]-out_df[all_appliances].sum(axis=1)
    out_df[out_df["Others"]<0] = 0
    out_df = out_df.drop("aggregate",1)
    out_df = out_df.rename(columns=appliance_name_map)
    out = out_df.to_dict('list')
    out['percentile'] = stats.percentileofscore(global_df_aggregate,dfc.ix[home_id][["aggregate_%d" %i for i in range(1, 13)]].sum() )
    return out

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index2.html')



@app.route('/query', methods=['POST'])
def query():
    data = json.loads(request.data)
    print data
    import flask.json as fjson
    out_data = predict(data["Occupants"], data["Rooms"],data["Area"],np.random.choice(all_homes['hvac'],1).tolist()[0])
    print out_data
    return fjson.jsonify(**out_data)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
