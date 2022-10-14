import numpy as np
import pandas as pd
import os
import csv

root_path = 'E:\dataset\\bikedata\\'
raw_data_path = "E:\dataset\\bikedata\\raw_data\\"
station_info_dir = 'E:\\dataset\\bikedata\\station_info\\'
record_path = 'E:\dataset\\bikedata\\sted_records\\'

# columns
# 'tripduration', 'starttime', 'stoptime', 'start station id',
#        'start station name', 'start station latitude',
#        'start station longitude', 'end station id', 'end station name',
#        'end station latitude', 'end station longitude']


# station info in each changed month
def findStationinfo():
    # INPUT: raw data
    # OUTPUT: station id, station lat, station lon, whether_new into date.csv
    print('find statin infor at each timestamp')
    files = os.listdir(raw_data_path)
    merge_st = pd.DataFrame({'station id': [], 'station latitude': [], 'station longitude': [],'build time':[],'wether_new':0.0})

    new_flag =0.0
    for f in files:
        print(f)
        date = f.split('-')[0]
        f_name = raw_data_path+f
        # collect start stations and end stations to find all station info
        df_i = pd.read_csv(f_name,header=0)

        if 'start station id' not in df_i.columns:
            df_i = df_i.rename(columns={'Start Station ID':'start station id','Start Station Latitude':'start station latitude','Start Station Longitude':'start station longitude', 'Start Time':'starttime',
                                 'End Station ID':'end station id','End Station Latitude':'end station latitude','End Station Longitude':'end station longitude','Stop Time':'stoptime'})
            print(df_i.columns)

        df_s = pd.DataFrame({'station id':df_i['start station id'].values,'station latitude':df_i['start station latitude'].values,'station longitude':df_i['start station longitude'].values,'build time':df_i['starttime'],'wether_new':new_flag})
        df_e =pd.DataFrame({'station id':df_i['end station id'].values,'station latitude':df_i['end station latitude'].values,'station longitude':df_i['end station longitude'].values,'build time':df_i['stoptime'],'wether_new':new_flag})
        df_s = df_s.append(df_e,ignore_index=True)

        # current station set
        df_st_date = df_s.drop_duplicates('station id',inplace=False)

        his_set_stid = set(merge_st['station id'].values)
        current_set_stid = set(df_st_date['station id'].values)
        diff_set = current_set_stid-his_set_stid


        # if station set has different stations save union set of two sets
        if len(diff_set)>0:
            merge_st = merge_st.append(df_st_date,ignore_index=True)
            merge_st.drop_duplicates('station id','first',inplace=True)

            # merge_st[merge_st['station id'] in diff_set]['wether_new'] = 1.0
            w_f_n = root_path + 'station_info\\' + date + '.csv'
            merge_st.to_csv(w_f_n, index=False)

            # new station convert to old station
            merge_st['wether_new'] = 0.0
            new_flag = 1.0


def cleanStationInfo():
    # INPUT: all station infor files
    # OUTPUT: station infor files without 0 loc
    print('clean station with 0 lat and longitude')
    files = os.listdir(station_info_dir)
    for f in files:
        f_name = station_info_dir+ f
        df = pd.read_csv(f_name)
        df_clean = df.drop(df[(df['station latitude']==0) | (df['station id']==0) | (df['station latitude']>45)].index)
        df_clean.to_csv(f_name,index=False)


# find the max and min of lar and lon
def findMaxminLoc():
    # INPUT: latest file of station info
    # OUTPUT: maximam and minimum of lat and long

    f_name = station_info_dir+'201810.csv'
    st_df = pd.read_csv(f_name)
    lat_min = st_df['station latitude'].values.min()
    lat_max = st_df['station latitude'].values.max()
    long_min = st_df['station longitude'].values.min()
    long_max = st_df['station longitude'].values.max()

    print(lat_min,lat_max,long_min,long_max)


# divide the city into regular grid
def dividintoGrid(min_lat,max_lat,min_long,max_long):
    # left long is smaller
    # top lat is larger
    # start 0 from top left
    # INPUT: lat and long of region
    # OUTPUT: grid id, top left node, bottom right node
    print('divide city into grids')
    lat_unit = 1/111
    long_unit = 1/55
    delta_long = max_long-min_long
    delta_lat = max_lat - min_lat

    num_along_long = delta_long/long_unit
    num_along_lat = delta_lat/lat_unit
    lat_i = max_lat
    long_i = min_long
    grid_id = 0

    w_f = root_path + 'grid\\'+'grid_info.csv'
    csv_w = csv.writer(open(w_f,'a',newline=''))
    cols = ['grid id','top_left_lat','top_left_long','bot_right_lat','bot_right_long']
    csv_w.writerow(cols)

    while lat_i > min_lat:
        while long_i <max_long:
            top_left_lat = lat_i
            top_left_long = long_i
            bot_right_lat = lat_i - lat_unit
            bot_right_long = long_i + long_unit

            region_v = [grid_id, top_left_lat,top_left_long,bot_right_lat,bot_right_long]
            csv_w.writerow(region_v)

            grid_id += 1
            long_i += long_unit
        lat_i -= lat_unit
        long_i = min_long



# assign  station into grids
def assignStation2Grids(st_df,grid_df):
    # INPUT: station infor, grid infor
    # OUTPUT: station infor with grid id

    grid_id = []
    loc = st_df[['station latitude','station longitude']].values
    for loc_i in loc:
        st_id = searchId(loc_i[0],loc_i[1],grid_df)
        grid_id.append(st_id)

    st_df['grid id'] = grid_id
    st_df.to_csv('E:\dataset\\bikedata\\grid\\st2grid.csv',index=False)


def searchId(lat, lon,grid_df):
    # left long is smaller
    # top lat is larger
    # print(grid_df['top_left_lat'])
    node2area = grid_df[(grid_df['top_left_lat']>lat)&(grid_df['top_left_long']<lon)&(grid_df['bot_right_lat']<lat)&(grid_df['bot_right_long']>lon)]
    res =  node2area['grid id'].values
    if res.size==0:
        return -1
    else:
        return res[0]


def findChangesinGrid():
    # find new station in grids
    # columns: date
    # row grid,
    # unit number of new stations

    st2grid = pd.read_csv('E:\dataset\\bikedata\\grid\\st2grid.csv')
    files = os.listdir(station_info_dir)

    grid_id_set = np.arange(0,263)
    chg_grids = pd.DataFrame({'grid id':grid_id_set})

    for f in files:
        date = f.split('.')[0]
        cur_new = np.zeros_like(grid_id_set)
        f_name = station_info_dir+f
        df_i = pd.read_csv(f_name,header=0)
        new_st = df_i[df_i['wether_new']==1]['station id'].values
        gn_id =st2grid[st2grid['station id'].isin(new_st)]['grid id'].values

        values, counts = np.unique(gn_id,return_counts=True)

        # how many stations do we add in each grid
        cur_new[values] =counts
        chg_grids[date] = cur_new

    chg_grids.to_csv('E:\dataset\\bikedata\\grid\\varied_grid.csv',index=False)


def addGrid4Raw(st2grid_df,time_head_st,time_head_et):
    # INPUT: raw data， st2grid
    # OUTPUT: time id, loc
    raw_fs = os.listdir(raw_data_path)
    for f in raw_fs:
        print('file name',f)
        f_name = raw_data_path+f
        df_i = pd.read_csv(f_name,header=0)
        # print(df_i.shape)
        df_i = df_i.dropna(axis=0,how='any')
        # print(df_i.shape)
        if 'start station id' not in df_i.columns:
            df_i = df_i.rename(columns={'Start Time':'starttime','Stop Time':'stoptime','Start Station ID':'start station id','Start Station Latitude':'start station latitude','Start Station Longitude':'start station longitude',
                                 'End Station ID':'end station id','End Station Latitude':'end station latitude','End Station Longitude':'end station longitude'})

        sm_df = df_i[['starttime','stoptime','start station id','start station latitude','start station longitude','end station id','end station latitude', 'end station longitude']]

        start_st_id= df_i['start station id'].values
        # print(start_st_id.shape)
        ed_st_id= df_i['end station id'].values
        st_g_id_set = []
        ed_g_id_set = []
        for i in range(len(start_st_id)):
            st_id = start_st_id[i]
            ed_id = ed_st_id[i]
            st_g = st2grid_df[st2grid_df['station id']==st_id]['grid id'].values
            ed_g = st2grid_df[st2grid_df['station id'] == ed_id]['grid id'].values
            if len(st_g)==0 or len(ed_g)==0:
                st_g = [-1]
                ed_g=[-1]

            st_g_id_set.append(st_g[0])
            ed_g_id_set.append(ed_g[0])


        sm_df['start grid'] = st_g_id_set
        sm_df['end grid'] = ed_g_id_set
        # sm_df.drop(sm_df[sm_df['start grid']==-1].index,inplace=True)
        df_g = sm_df[sm_df['start grid']!=-1]
        df_g.reset_index(inplace=True)
        print(df_g)

        print('sm_df.shape:',df_g.shape)

        s_time_df = separatetime(df_g, time_head_st)
        s_time_df.reset_index(inplace=True)
        e_time_df = separatetime(df_g, time_head_et)
        e_time_df.reset_index(inplace=True)
        print('s_time_shape:',s_time_df.shape)
        print('e_time_shape:',e_time_df.shape)

        m_df = pd.concat([df_g,s_time_df,e_time_df],axis=1)
        print('merge_df_shape',m_df.shape)

        w_f = root_path+'sted_records\\' + f.split('-')[0] +'.csv'
        m_df.to_csv(w_f,index=False)



def separatetime(df,col_name):
    # print(df.columns)
    date_list = df[col_name].values
    day_time = []
    i = 0
    for d in date_list:
        day,time = d.split()
        time = int(time.split(":")[0])
        day_time.append([day,time])
        i += 1
    # print(day_time)
    n_col_names =[]
    if 'tart' in col_name or 'pick' in col_name:
        n_col_names=['start day','start stmp']
    elif 'stop' in col_name or 'drop' in col_name:
        n_col_names=['end day','end stmp']
    else:
        n_col_names=['date','time']
    time_df = pd.DataFrame(day_time, columns=n_col_names)
    # print(time_df)
    return time_df

# def searchArea(df):
#
#     value = df.values
#     grid_assign =[]
#
#     for row in value:
#         start_lat = float(row[5])
#         start_lon = float(row[6])
#         end_lat = float(row[9])
#         end_lon = float(row[10])
#         start_grid_id = searchId(start_lat,start_lon)
#         end_grid_id = searchId(end_lat,end_lon)
#         n_row = np.append(row,[start_grid_id,end_grid_id])
#         grid_assign.append(n_row)
#     cols = df.columns.values
#     cols =np.append(cols,['start_grid','end_grid'])
#     df_grid = pd.DataFrame(data=grid_assign,columns=cols)
#     df_grid.to_csv('E:\\dataset\\shareST\\bike\\bike201804.csv',index=False)


def assignDemds2Grid(f_name,gp_cols):
    # INPUT: raw data,['start day','start time','start grid']
    # output demands in each grid at each hour
    # time span should be controlled
    # column: timestamp
    # row: girds
    df = pd.read_csv(f_name)
    df['count'] = 1.0
    select_df = df[gp_cols]

    select_gp = select_df.groupby(gp_cols[:3]).sum()
    select_gp = select_gp.reset_index()
    select_gp = select_gp.sort_values(by=gp_cols[:3])

    print(select_gp)
    print(select_gp[select_gp['start day']=='9/1/2015'])



# STEP 1
# findStationinfo()

# STEP 2
# cleanStationInfo()

# STEP 3
# findMaxminLoc()

# STEP 4
# mm_loc_df = pd.read_csv('E:\dataset\\bikedata\\min_max_loc.csv',header=0)
# min_lat= mm_loc_df['min lat'].values[0]
# max_lat = mm_loc_df['max lat'].values[0]
# min_long = mm_loc_df['min long'].values[0]
# max_long = mm_loc_df['max long'].values[0]
# dividintoGrid(min_lat,max_lat,min_long,max_long)

# STEP 5
# latest_st_fname = station_info_dir + '201810.csv'
# st_df = pd.read_csv(latest_st_fname)
# grid_fname = root_path+'grid\\'+'grid_info.csv'
# grid_df = pd.read_csv(grid_fname,header=0)
# assignStation2Grids(st_df,grid_df)

# STEP 6
# st2grid_df = pd.read_csv('E:\dataset\\bikedata\\grid\\st2grid.csv')
# addGrid4Raw(st2grid_df,'starttime','stoptime')



# STEP 7 collect demands in each grid

gp_cols =  ['start day','start stmp','start grid','count']
files = os.listdir(record_path)
for f in files:
    print(f)
    f_name = record_path+f
    assignDemds2Grid(f_name,gp_cols)
    break
# f_name = raw_data_path+'201611-citibike-tripdata.csv'
# df = pd.read_csv(f_name)
# n_df = df.dropna(axis=0,how='any')
# print(df.shape)
# print(n_df.shape)