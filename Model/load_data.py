import numpy as np
import pylab as p
from sklearn.preprocessing import MinMaxScaler


def load_train_data(source_data,seq,inorout):
    data =[]
    if inorout =='in':
        data = source_data[:,:,:,0]
    elif inorout =='out':
        data = source_data[:,:,:,1]
    data = np.expand_dims(data, -1)
    train_p = int(0.7*np.shape(data)[1])
    train_set = data[:,:train_p,:,:] #[n,grid,seq,1]
    
    x_train = train_set[:,:,:seq,:]
    y_train = train_set[:,:,seq,:]
    return x_train,y_train




def load_test_data(source_data,inorout):
    data = []
    if inorout =='in':
        data = source_data[:,:,:,0]
    elif inorout =='out':
        data = source_data[:,:,:,1]
    data = np.expand_dims(data, -1)
    test_p = int(0.3* np.shape(data)[1])
    test_set = data[:,-test_p:,:,:]
    return test_set


def load_train_env_data(source_data,env_n,inorout):
    data = []
    if inorout =='in':
        data = source_data[:,:,:,0]
    elif inorout =='out':
        data = source_data[:,:,:,1]
    data = np.expand_dims(data,-1)
    train_p = int(0.7 * np.shape(data)[1])
    # divide to two balanced datasets
    train_set =[]
    unit = int(train_p/env_n)
    st_i=0
    ed_i = unit
    for i in range(env_n):
        train_set.append(data[:,st_i:ed_i,:,:]) #[ts,g,his,v]
        st_i = ed_i
        ed_i= ed_i+unit
    return train_set

def load_bike_train(source_data,env_n,inorout,batchsize):
    if inorout =='in':
        data = source_data['traffic'][:7913,:,:,0]
    elif inorout =='out':
        data = source_data['traffic'][:7913,:,:,1]

    data = np.expand_dims(data, -1)
    train_p = int(0.3 * np.shape(data)[1])
    his_len = np.shape(data)[2]
    # divide to two balanced datasets
    train_set = []
    unit = int(train_p / env_n)
    st_i = 0
    ed_i = unit
    train_all_length = []
    for i in range(env_n):
        env_data = np.reshape(data[:, st_i:ed_i, :, :],[-1,his_len,1])
        train_set.append(env_data)  # [ts,g,his,v]
        st_i = ed_i
        ed_i = ed_i + unit
        train_all_length.append(np.shape(env_data)[0])

    train_set_slice=[]
    while True:
        for e in range(env_n):
            batchid_e = select_index(train_all_length[e], batchsize)
            batch_e_data = train_set[e][batchid_e]
            train_set_slice.append(batch_e_data)
        yield train_set_slice
        train_set_slice=[]

def load_bike_test(source_data,inorout):
    if inorout =='in':
        data = source_data['test'][:,:,:,0]
    elif inorout =='out':
        data = source_data['test'][:,:,:,1]
    test_data = np.expand_dims(data, -1)
    return test_data

def select_index(upperbound,batchsize):
    batch_id = np.random.choice(upperbound,batchsize)
    return batch_id

# def load_wph_train(dir,seq_len,env_n,batchsize):
#     # dir = 'E:\dataset\Data\WATERPH/train_test/'
#     # train_f_name = ['environment_midwest','environment_northeast','environment_pacific','environment_south','environment_southwest']
#     train_f_name =['environment_west']
#     train_all_set = []
#     train_all_length = []
#     if len(train_f_name)==1:
#         f_e = dir+train_f_name[0]+'s'+str(seq_len)+'.npz'
#         data_e = np.load(f_e)['environment']
#         len_e = np.shape(data_e)[0]
#         train_p = int(0.9* len_e)

#         # first 42 years
#         train_all_set.append(data_e[:train_p])
#         train_all_length.append(train_p)
#     else:
#         for e in range(env_n):
#             f_e = dir+train_f_name[e]+'s'+str(seq_len)+'.npz'
#             data_e = np.load(f_e)['environment']
#             len_e = np.shape(data_e)[0]
#             train_p = int(0.9* len_e)

#             # first 42 years
#             train_all_set.append(data_e[:train_p])
#             train_all_length.append(train_p)
        
    
#     train_set = []

#     while True:
#         for e in range(env_n):
#             batchid_e = select_index(train_all_length[e], batchsize)
#             batch_e_data = train_all_set[e][batchid_e]
#             train_set.append(batch_e_data)
#         yield train_set
#         train_set=[]

# original
def load_wph_train(dir,seq_len,env_n,batchsize):
    # dir = 'E:\dataset\Data\WATERPH/train_test/'
    train_f_name = ['environment_midwest','environment_northeast','environment_pacific','environment_south','environment_southwest']
    train_all_set = []
    train_all_length = []
    
    for e in range(env_n):
        f_e = dir+train_f_name[e]+'s'+str(seq_len)+'.npz'
        data_e = np.load(f_e)['environment']
        len_e = np.shape(data_e)[0]
        train_p = int(0.9* len_e)

        # first 42 years
        train_all_set.append(data_e[:train_p])
        train_all_length.append(train_p)
        
    
    train_set = []

    while True:
        for e in range(env_n):
            batchid_e = select_index(train_all_length[e], batchsize)
            batch_e_data = train_all_set[e][batchid_e]
            train_set.append(batch_e_data)
        yield train_set
        train_set=[]

def load_wph_test(dir,seq_len,span):
    # dir = 'E:\dataset\Data\WATERPH/train_test/'
    # 30*25*12=9000 last one year
    test_f_name = ['environment_west']
    test_f = dir+test_f_name[0]+'s'+str(seq_len)+'.npz'
    data = np.load(test_f)['environment']
    dur_e = int((5-span)*9000)
    dur_s = int(dur_e+9000)
    test_data = data[-dur_s:-dur_e]
    return test_data

# def load_test_data(data):
#
#     test_set1 = data[5000:6000]
#     test_set2 = data[6000:7000]
#     test_set3 = data[7000:8000]
#     return test_set1,test_set2,test_set3

def inverse_transform(path, city,data,inorout):
    if city=='water':
        mm_v = [14.,0.]
    else:
        dir = path + city + 'mm.npz'
        tag = 'maxmin_' + inorout
        mm_v = np.load(dir)[tag]

    # norm: data-min/(max-min)
    inv_data = (data*(mm_v[0]-mm_v[1]))+mm_v[1]
    return inv_data

