import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# have zero
bike_path = 'E:\\dataset\\Data\\Data\\bike\\'
# bike files: chicago, nyc(10*20), dc(16*16)

taxi_path = 'E:\\dataset\\Data\\Data\\taxi\\'
# taxi files: chicago, nyc, dc

def findNSgrid(data,r_no,c_no):
    ns_grid_set = []
    for i in range(r_no):
        for j in range(c_no):
            one_grid_data = data[:, i, j, 0]
            sum_g =  np.count_nonzero(one_grid_data)
            if sum_g>np.shape(data)[0]*0.6:
                print('grid_i,grid_j',i,j)
                ns_grid_set.append([i,j])
                print('sum of demands in grid:',sum_g)
    return ns_grid_set


def visData(city,g_i,g_j):
    # visulize data in non-sparse grid
    print('Data in',city)
    file_name = bike_path +city+'.npz'
    datas = np.load(file_name)
    # content in datas:
    # print(datas.files)
    data = datas['traffic']
    print('data shape [stmp, g_r,g_c, in/out]', np.shape(data))
    print('---')
    # demands in one grid
    sp_g_data = data[:,g_i,g_j,0]
    # hourly demands
    plt.subplot(2, 1, 1)
    sns.lineplot(np.arange(len(sp_g_data)), sp_g_data)
    plt.subplot(2, 1, 2)
    sns.lineplot(np.arange(720),sp_g_data[:720])
    sns.lineplot(np.arange(720),sp_g_data[2160:2880])


    plt.show()

def filteData(dir, city,w_dir,g_r,g_c):
    # find
    # input: raw data array
    f_name = dir+city+'.npz'
    data = np.load(f_name)['traffic']
    grid_set = findNSgrid(data,g_r,g_c)

    filer_demand=[]
    for g in grid_set:
        filer_demand.append(data[:,g[0],g[1],:])

    arr = np.transpose(np.array(filer_demand),[1,0,2])


    w_f = w_dir + city+'.npz'
    np.savez(w_f,traffic=arr)

def norm_data(path,city):
    f_name = path+city+'.npz'
    data = np.load(f_name)['traffic']
    in_data = data[:,:,0]
    out_data = data[:,:,1]

    # norm data
    in_max,in_min,in_norm = minmaxTransform(in_data)
    out_max, out_min,out_norm= minmaxTransform(out_data)

    norm_f = path+city+'norm.npz'
    traffic = np.stack([in_norm,out_norm],axis=-1)
    np.savez(norm_f, traffic=traffic)
    mm_f = path+city+'mm.npz'
    np.savez(mm_f,maxmin_in=[in_max,in_min],maxmin_out = [out_max,out_min])


def minmaxTransform(data):
    i_max = np.max(data)
    i_min = np.min(data)
    norm_data = (data-i_min)/(i_max-i_min)
    return i_max,i_min,norm_data



def segData(data,seq_len):
    # input raw_data_f_name, seq_len,
    # output:[t0,t1,t2]->t3, [t1,t2,t3]->t4
    # data = np.load(f_name)['in_data']
    ed_i=seq_len
    st_i =0
    seq_set =[]
    while ed_i < len(data)-1:
        part = data[st_i:ed_i+1] #[seq_len+1, g, v]
        seq_set.append(part)
        st_i +=1
        ed_i+=1

    arr = np.transpose(np.array(seq_set),[0,2,1,3]) #[l-seq,seq,g,v] ->[l-seq,g,seq,v]
    return arr

def seg_stdiff_data(data,seq_len,span,w_2_f):
    # input raw_data_f_name([ts,grid,]), seq_len, span(weeks between train and test)
    # split train and test duration(last 14 days-->336 hours)
    # output 2 separate files:train: [t0,t1,t2]->t3, test [t2+span,t3+span,t4+span]->t5+span
    # the length of train/test set for each span is the same

    test_ts_length = 7 * 24
    span_ts = span*7*24
    longest_span_ts = 4*7*24

    # train sample num
    train_sample_num = int(0.7 * np.shape(data)[1])
    train_ts_length = len(data) - test_ts_length-longest_span_ts
    # extract train data data[all_ts-test_ts-span,0.7*all]
    train_data = data[:train_ts_length, :train_sample_num, :]
    print(np.shape(train_data))


    # test length
    test_sample_num = int(0.3 * np.shape(data)[1])
    test_duration_st = train_ts_length+span_ts
    test_duration_ed = test_duration_st+test_ts_length
    test_data = data[test_duration_st:test_duration_ed,-test_sample_num:,:]
    print(np.shape(test_data))

    train_seg_data = segData(train_data,seq_len)
    print(np.shape(train_seg_data))

    test_seg_data = segData(test_data,seq_len)
    print(np.shape(test_seg_data))
    np.savez(w_2_f, train =train_seg_data,  test=test_seg_data)




# visData('nyc',6,3)


# w_path = 'E:\\dataset\\Data\\Data\\filter_bike\\'
# filter zero grid
# filteData(bike_path,'dc',w_path,16,16)

# segment data into slice
# normAseg_data(w_path,'nyc',seq_len=12)

# norm_data
# norm_data(w_path,'nyc')

#  load norm data
city = 'dc'
seq = 6
read_from_f = 'E:\dataset\Data\Data\\filter_bike\\'+city+'norm.npz'
norm_data_arr = np.load(read_from_f)['traffic']

# seg_data only by spatial: same time duration but different stations
# seg_arr = segData(norm_data_arr,seq_len=seq)
# w_2_f = 'E:\dataset\Data\Data\\filter_bike\\'+city+str(seq)+'.npz'
# np.savez(w_2_f,traffic=seg_arr)

# seg data into diff
# span range from 0 week to 4 weeks
# span_d = 4
# w_2_f_span = 'E:\dataset\Data\Data\\filter_bike\\'+city+'l'+str(seq)+'s'+str(span_d)+'.npz'
# seg_stdiff_data(norm_data_arr,seq,span_d, w_2_f_span)
