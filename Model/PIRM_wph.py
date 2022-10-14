import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import optimizers
from datetime import datetime as dt
from load_data import load_wph_train, inverse_transform, load_wph_test
from EnvConfounderIRM import EnvAware

path = '/data/user18100643/dataset/water/'
city='water'
inorout='water'
emb_size = 32
seq_len = 12
steps = 50
batchsize = 128
batch_num = 2500
learning_rate = 1e-3
div_en_n = 3  # the number of environments maximun=5
span=1
current_time = dt.now().strftime("%Y%m%d-%H%M%S")
reg_weight = 0
irm_weight_init = 10
contrs_weight_init = 10

# physical_devices = tf.config.list_physical_devices('GPU')
# # print('available devuces:',physical_devices)
# tf.config.experimental.set_visible_devices(physical_devices[3], device_type='GPU')
# tf.config.experimental.set_memory_growth(physical_devices[3], enable=True)



# log files  
log_dir = '../logs/'+city+'/gradient_tape/'  + current_time + '/PIRM'  +'-span'+str(span)+str(irm_weight_init)+'-'+str(contrs_weight_init)+'-'+str(learning_rate)+'seq'+str(seq_len)+'emb'+str(emb_size)
train_loss = tf.keras.metrics.Mean(name='train_mse')
train_real_mse = tf.keras.metrics.Mean(name='train_real_mse')
# test_mse_err = tf.keras.metrics.Mean(name='test_mse')
# test_mae_err = tf.keras.metrics.Mean(name='test_mae')
result_summary_writer = tf.summary.create_file_writer(log_dir)

# initialize model
MODEL = EnvAware(emb_size)

optimizer = optimizers.Adam(learning_rate=learning_rate)
loss_func = tf.keras.losses.MeanSquaredError()


def train_step(train_env_set, div_en_n,rec_flg,irm_weight,contrs_weight):
    # loss_y and loss_irm in each env
    # loss_c between envs
    env_label_set = []
    irm_feature_set = []
    env_feature_set = []
    loss_y_env = []
    penalty_env = []
    outputs_env = []
    y_true_env = []
    with tf.GradientTape(persistent=True) as g:
        for e in range(div_en_n):
            train_x = tf.convert_to_tensor(train_env_set[e][:, :seq_len, :], dtype=tf.float32)
            train_y = tf.convert_to_tensor(train_env_set[e][:, seq_len, :], dtype=tf.float32)
            outputs, irm_features, env_features = MODEL(train_x)

            outputs_env.append(outputs)
            y_true_env.append(train_y)

            # environment label [0,..,0,1,...,1]
            env_label_set.append(tf.ones(shape=tf.shape(outputs)[0]) * e)
            env_feature_set.append(env_features)
            irm_feature_set.append(irm_features)

            MSE_loss = loss_func(train_y, outputs)
            with tf.GradientTape() as gg:
                irm_loss, scale = MODEL.loss_irm(train_y, outputs)
            scale_grads = gg.gradient(irm_loss, [scale])
            penalty = tf.pow(scale_grads, 2)

            penalty_env.append(penalty)
            loss_y_env.append(MSE_loss)

        # mean for mse loss and penalty in each env
        loss_y = tf.reduce_mean(tf.stack(loss_y_env))
        loss_irm = tf.reduce_mean(tf.stack(penalty_env))

        # regulazation
        l2_reg = tf.reduce_mean([tf.nn.l2_loss(v) for v in MODEL.trainable_variables])
        # y_loss = MSE_loss+ reg_weight * l2_reg

        loss_contrastive = MODEL.loss_c(env_feature_set, env_label_set)

        loss4irm = loss_y + irm_weight * loss_irm
        # if irm_weight > 1.:   loss4irm /= irm_weight

        loss4contrs = loss_y + contrs_weight * loss_contrastive
        # if contrs_weight > 1.: loss4contrs /= contrs_weight

        loss4pred = loss_y + contrs_weight * loss_contrastive + irm_weight * loss_irm
        if irm_weight > 1. or contrs_weight > 1.:
            trade_max = np.max([irm_weight, contrs_weight])
            loss4pred /= trade_max

    grads4irm = g.gradient(loss4irm, MODEL.get_env_irm_v())
    grads4contras = g.gradient(loss4contrs, MODEL.get_env_rel_v())
    grads4pred = g.gradient(loss4pred, MODEL.get_out_v())

    optimizer.apply_gradients(zip(grads4irm, MODEL.get_env_irm_v()))
    optimizer.apply_gradients(zip(grads4contras, MODEL.get_env_rel_v()))
    optimizer.apply_gradients(zip(grads4pred, MODEL.get_out_v()))




    if rec_flg:
        res_rec_f = '../res/train_emb_res'+current_time+'.npz'
        np.savez(res_rec_f,irm_feature=irm_feature_set,env_feature = env_feature_set)

    return tf.concat(outputs_env, axis=0), tf.concat(y_true_env, axis=0), loss4pred


def test_step(test_x, test_y,rec_flg):
    pred,irm_feat_test,env_feat_test = MODEL.predict(test_x)

    # inverse transform
    real_pred = inverse_transform(path, city, pred, inorout)
    real_label = inverse_transform(path, city, test_y.numpy(), inorout)
    real_mse = mean_squared_error(real_label, real_pred)
    real_mae = mean_absolute_error(real_label, real_pred)
    nonz_id = np.where(real_label!=0)
    diff = np.abs(real_label[nonz_id]-real_pred[nonz_id])/real_label[nonz_id]
    real_mape = np.sum(diff)/np.size(real_label)

    if rec_flg:
        res_rec_f = '../res/test_emb_res' + current_time + '.npz'
        np.savez(res_rec_f, irm_feature=irm_feat_test, env_feature=env_feat_test)

    return real_mse, real_mae, real_mape


def test(s,rec_flg):
    # test_pred = []
    test_set = load_wph_test(path,seq_len,span)
    # for t_s in range(len(test_set)):
    #     test_x_tensor = tf.convert_to_tensor(test_set[t_s, :seq_len, :], dtype=tf.float32)
    #     test_y_tensor = tf.convert_to_tensor(test_set[t_s, seq_len, :], dtype=tf.float32)
    #     mse, mae = test_step(test_x_tensor, test_y_tensor)
    #     # record prediction result at each timestamp
    #     # test_pred.append(mse1)
    #
    #     # mean of mse and mae for all timestamps
    #     test_mse_err(mse)
    #     test_mae_err(mae)
    test_x_tensor = tf.convert_to_tensor(test_set[:, :seq_len, :], dtype=tf.float32)
    test_y_tensor = tf.convert_to_tensor(test_set[:, seq_len, :], dtype=tf.float32)
    mse, mae,mape = test_step(test_x_tensor,test_y_tensor,rec_flg)

    rmse1 = np.sqrt(mse)
    print('RMSE for test set 1:', rmse1)
    print('MSE for test set 1:', mse)
    with result_summary_writer.as_default():
        tf.summary.scalar(name='test_rmse', data=rmse1, step=s)
        tf.summary.scalar(name='test_mse', data=mse, step=s)
        tf.summary.scalar(name='test_mae', data=mae, step=s)
        tf.summary.scalar(name='test_mape', data=mape, step=s)



loss_e = []
# penalty_e = []
# TRAIN
# get same num of samples from each environment
generator = load_wph_train(path,seq_len,div_en_n,batchsize)

record_flg=0.
for s in range(steps):
    if s<5:
        irm_weight = 0.
        contrs_weight = 0.
    else:
        irm_weight =  irm_weight_init
        contrs_weight = contrs_weight_init
    for batch in range(batch_num):
        if s == steps - 1 and batch ==batch_num - 1: record_flg = 1.
        tr_set = next(generator)
        

        outputs, y_true, loss_t = train_step(tr_set, div_en_n, record_flg,irm_weight,contrs_weight)

        real_pred = inverse_transform(path, city, outputs.numpy(), inorout)
        real_label = inverse_transform(path, city, y_true.numpy(), inorout)
        real_mse = mean_squared_error(real_label, real_pred)
        print('step:', s, 'sample:', batch, 'train_mse:', loss_t)

        train_real_mse(real_mse)
        train_loss(loss_t)

    with result_summary_writer.as_default():
        tf.summary.scalar(name='train_env_mean_loss', data=train_loss.result(), step=s)
        tf.summary.scalar(name='train_real_mse', data=train_real_mse.result(), step=s)

    train_loss.reset_state()
    train_real_mse.reset_state()

    test(s,record_flg)
    print('=======================')

result_summary_writer.close()



