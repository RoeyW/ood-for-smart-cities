import tensorflow as tf

import tensorflow as tf
import losses


class EnvAware(tf.keras.Model):
    def __init__(self, emb_dim):
        super(EnvAware, self).__init__()
        self.emb_dim = emb_dim

        # irm to learn environment invariant features
        self.irm_layers = self.env_irm_feature_extracter()
        # self.irm_layers = tf.keras.Sequential()
        # self.irm_layers.add(tf.keras.layers.GRU(emb_dim, return_sequences=True))
        # self.irm_layers.add(tf.keras.layers.GRU(emb_dim))

        # env confounder to learn environment related features
        self.envcf_layers = self.env_rel_feature_extracter()
        # self.envcf_layers = tf.keras.Sequential()
        # self.envcf_layers.add(tf.keras.layers.GRU(emb_dim, return_sequences=True))
        # self.envcf_layers.add(tf.keras.layers.GRU(emb_dim))

        # predict
        self.output_layer = self.output_regrs()
        # self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        self.MSE_loss = tf.keras.losses.MeanSquaredError()

    def _get_trainable_variables(self,m):
        # get trainable variables in model m
        model_vars = m.trainable_variables
        return model_vars

    def env_irm_feature_extracter(self):
        return tf.keras.Sequential([tf.keras.layers.GRU(self.emb_dim, return_sequences=True),
                                   tf.keras.layers.GRU(self.emb_dim)]
        )

    def env_rel_feature_extracter(self):
        return tf.keras.Sequential([tf.keras.layers.GRU(self.emb_dim, return_sequences=True),
                                   tf.keras.layers.GRU(self.emb_dim)]
        )

    def output_regrs(self):
        return tf.keras.Sequential([tf.keras.layers.Dense(self.emb_dim),
            tf.keras.layers.Dense(1)])


    def get_env_irm_v(self):
        return self._get_trainable_variables(self.irm_layers)

    def get_env_rel_v(self):
        return self._get_trainable_variables(self.envcf_layers)

    def get_out_v(self):
        return self._get_trainable_variables(self.output_layer)


    def loss_y(self,y_true,y_pred):
        return self.MSE_loss(y_true,y_pred)


    def loss_irm(self,y_true,y_pred):
        scale = tf.Variable([1.0],trainable=True)
        y_scale = y_pred*scale
        return self.MSE_loss(y_true,y_scale),scale

    def loss_c(self,features, env_label):
        # compare different environment
        # loss fnc from <supervised contrastive learning>
        # features: [env_size, batch_size, feature_dim]
        # env_label: [env_size,batch_size]

        # features->[env_size*batch_size]
        features_com = tf.concat(features,axis=0)
        env_label_com = tf.concat(env_label,axis=0)


        contrastive_loss = losses.supervised_nt_xent_loss(features_com,env_label_com)
        return contrastive_loss


    def __call__(self, inputs, **kwargs):
        # [batch, timestmp, ]
        irm_emb = self.irm_layers(inputs)
        envcf_emb = self.envcf_layers(inputs)
        fuse_emb = tf.concat([irm_emb,envcf_emb],axis=-1)
        output = self.output_layer(fuse_emb)

        return output,irm_emb,envcf_emb
