import math

import numpy as np
import tslearn.utils
from scipy.spatial import distance
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
import collections
import tslearn.clustering as ts_c
from dataprocess.segdata import segData

# different methods: dtw, euclidean
path = 'E:\\dataset\\Data\\Data\\'
city='nyc'
data_source = 'filter_bike'
CLUSTER_N = 3
metric = 'euclidean'
seq_len=12

f_name = path + data_source +'\\'+ city+'norm.npz'
data = np.load(f_name)['traffic'] #[timestamp, g, 2]

def ts_sim(x,y,method):
    if method == 'cosine':
        return distance.cosine(x,y)
    if method == ' euclidean':
        return distance.euclidean(x,y)
    if method == 'pearsonr':
        return pearsonr(x,y)


def cluster_ts(data,n, m):
    ts_data = tslearn.utils.to_time_series_dataset(data)
    k_model = ts_c.TimeSeriesKMeans(max_iter_barycenter=50 ,n_clusters=n,metric=m,max_iter=50,random_state=0)
    k_model.fit(ts_data)
    ts_l = k_model.labels_
    return ts_l


def divd_train_test(source_data, n_c, sc_label):
    # sourcedata.shape = [ts,g,v]
    # reture [ts, g, v]
    stats_lable = dict(collections.Counter(sc_label))
    print(stats_lable)
    sc_label_all = list(stats_lable.keys())

    train_n = math.ceil(n_c *0.6)
    gfirst_source_data = np.transpose(source_data,[1,0,2])
    train_set = []
    test_set=[]
    i=0
    while i < train_n:
        ginc = np.transpose(gfirst_source_data[sc_label==sc_label_all[i]],[1,0,2])
        ginc_seg = segData(ginc,seq_len)
        train_set.append(ginc_seg)
        i+=1
    while i <n_c:
        ginc = np.transpose(gfirst_source_data[sc_label==sc_label_all[i]],[1,0,2])
        ginc_seg = segData(ginc, seq_len)
        test_set.append(ginc_seg)
        i+=1

    train_set_arr = np.concatenate(train_set,axis=1)
    test_set_arr = np.concatenate(test_set,axis=1)

    # erm train test file
    erm_path = 'E:\\dataset\\Data\\Data\\filter_bike\\'+city+'erm'+str(seq_len)+'.npz'
    np.savez(erm_path,train= train_set_arr,test = test_set_arr)

    # irm train test file
    irm_path = 'E:\\dataset\\Data\\Data\\filter_bike\\'+city+str(seq_len)+'irm.npz'
    np.savez(irm_path,train=train_set,test=test_set)







# pre_data = np.transpose(data[:720],[1,0,2])
# grid_n = np.shape(pre_data)[0]
# feat_data = pre_data[:,:,0]

# predict the cluster for each grid according to 24h demands
# ts_cli = cluster_ts(feat_data,CLUSTER_N,metric)
# print(ts_cli)

# write train_dataset and test dataset into file

# divd_train_test(data,CLUSTER_N,ts_cli)

# train_pre_seg_data = np.load('E:\\dataset\\Data\\Data\\filter_bike\\'+city+'erm.npz')
# train_slice_erm = segData(pre_seg_data['train'],seq_len)
# test_slice_erm = segData(pre_seg_data['test'],seq_len)
# erm_slice_w_f = 'E:\\dataset\\Data\\Data\\filter_bike\\'+city+'erm'+str(seq_len)+'.npz'
# np.savez(train=train_slice_erm,test=test_slice_erm)







