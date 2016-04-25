import pandas as pd
import glob
import os

HOURLY = False

if HOURLY:
    TIME_FIELD = "localhour"
else:
    TIME_FIELD = "local_15min"

FILES_PATH = "/if6/nb2cz/wiki_15min/"
STORE_PATH = "/if6/nb2cz/wiki-15min.h5"

# Dropping days when DST changed

dst_times = []
dst_times.append(('2012-03-11 02:00:00', '2012-03-11 03:00:00'))
dst_times.append(('2012-11-04 01:00:00', '2012-11-04 02:00:00'))
dst_times.append(('2013-03-10 02:00:00', '2013-03-10 03:00:00'))
dst_times.append(('2013-11-03 01:00:00', '2013-11-03 02:00:00'))
dst_times.append(('2014-03-09 02:00:00', '2014-03-09 03:00:00'))
dst_times.append(('2014-11-02 01:00:00', '2014-11-02 02:00:00'))
dst_times.append(('2015-03-08 02:00:00', '2015-03-08 03:00:00'))
dst_times.append(('2015-11-01 01:00:00', '2015-11-01 02:00:00'))




# Feeds to ignore
feed_ignore = ['gen', 'grid']

#list_of_buildings = glob.glob("/Users/nipunbatra/w/*.csv")
files = os.listdir(FILES_PATH)
file_size= {x:os.path.getsize(FILES_PATH+x) for x in  files if '.csv' in x}
file_series = pd.Series(file_size)
file_series = file_series.drop("dataport-metadata.csv")
fs = file_series[file_series>1000]

store = pd.HDFStore(STORE_PATH, mode='a', complevel=9, complib='blosc')
count = 0
for building_number_csv in fs.index:
    print "Done %d of %d" %(count, len(fs))
    try:
        building_path = os.path.join(FILES_PATH, building_number_csv)
        building_number = int(building_number_csv[:-4])
        if building_number in store.keys():
            continue
        df = pd.read_csv(building_path)
        df.index = pd.to_datetime(df[TIME_FIELD])
        df = df.drop(TIME_FIELD, 1)
        # Dropping feeds
        for feed in feed_ignore:
            if feed in df.columns:
                df = df.drop(feed, 1)

        df = df.mul(1000)

        # Dropping feeds with 0 sum
        cols_to_keep = df.sum()[df.sum()>0].index
        df = df[cols_to_keep]

        # Dropping dataid
        if "dataid" in df.columns:
            df = df.drop('dataid', 1)
        df = df.drop(['Unnamed: 0'], axis=1)


        # Fixing DST issues
        for start, end in dst_times:
            ix_drop = df[start:end].index
            df = df.drop(ix_drop)


        # Assigning local timezone
        df = df.tz_localize('US/Central')

        # Making data float32
        df = df.astype('float32')

        # Write in temp HDF5 store
        store.put(str(building_number), df, format='table')
        count = count + 1
    except Exception, e:
        print e
