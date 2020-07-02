import numpy as np
import tensorflow as tf
#from layer import dice
#from utils import sequence_mask

class DIN(object):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        # Embedding layer
        self.n_uid = n_uid
        self.n_mid = n_mid
        self.n_cat = n_cat
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.ATTENTION_SIZE = ATTENTION_SIZE
        
    def dice(self,_x, axis=-1, epsilon=0.000000001, name=''):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            alphas = tf.get_variable('alpha'+name, _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
            input_shape = list(_x.get_shape())

            reduction_axes = list(range(len(input_shape)))
            del reduction_axes[axis]
            broadcast_shape = [1] * len(input_shape)
            broadcast_shape[axis] = input_shape[axis]

        # case: train mode (uses stats of the current batch)
        mean = tf.reduce_mean(_x, axis=reduction_axes)
        brodcast_mean = tf.reshape(mean, broadcast_shape)
        std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
        std = tf.sqrt(std)
        brodcast_std = tf.reshape(std, broadcast_shape)
        x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
        # x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)
        x_p = tf.sigmoid(x_normed)
        
    def build_fcn_net(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        #dnn1 = self.dice(_x=dnn1, name='dice_1')
        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        #dnn2 = self.dice(dnn2, name='dice_2')
        dnn3 = tf.layers.dense(dnn2, 1, activation=None, name='f3')
        y_hat = tf.nn.sigmoid(dnn3)
        return y_hat
    
    def din_attention(self,query, facts, attention_size, mask, softmax_stag=1, return_alphas=False):
        mask = tf.equal(mask, tf.ones_like(mask))
        facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
        querry_size = query.get_shape().as_list()[-1]
        queries = tf.tile(query, [1, tf.shape(facts)[1]])
        queries = tf.reshape(queries, tf.shape(facts))
        din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
        d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att')
        d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')
        d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att')
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
        scores = d_layer_3_all
        # Mask
        # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
        key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
        paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

        # Activation
        if softmax_stag:
            scores = tf.nn.softmax(scores)  # [B, 1, T]

        # Weighted sum
        output = tf.matmul(scores, facts)  # [B, 1, H]
      
        return output

    def forward(self,features):
        uid = features['uid']
        #print(uid)
        target_mid = features['target_mid']
        target_cat = features['target_cat']
        prev_items = features['mid']
        prev_cats = features['mid_cat']
        mask = features['mask']
        mid_len = features['mid_len']
        
        with tf.name_scope('Embedding_layer'):
            uid_embeddings_var = tf.get_variable("uid_embedding_var", [self.n_uid, self.EMBEDDING_DIM])
            #tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
            uid_batch_embedded = tf.nn.embedding_lookup(uid_embeddings_var, uid)

            mid_embeddings_var = tf.get_variable("mid_embedding_var", [self.n_mid, self.EMBEDDING_DIM])
            #tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
            mid_batch_embedded = tf.nn.embedding_lookup(mid_embeddings_var,target_mid)
            mid_his_batch_embedded = tf.nn.embedding_lookup(mid_embeddings_var, prev_items)

            cat_embeddings_var = tf.get_variable("cat_embedding_var", [self.n_cat, self.EMBEDDING_DIM])
            #tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
            cat_batch_embedded = tf.nn.embedding_lookup(cat_embeddings_var, target_cat)
            cat_his_batch_embedded = tf.nn.embedding_lookup(cat_embeddings_var, prev_cats)

        item_eb = tf.concat([mid_batch_embedded, cat_batch_embedded], 1)
        item_his_eb = tf.concat([mid_his_batch_embedded, cat_his_batch_embedded], 2)
        item_his_eb_sum = tf.reduce_sum(item_his_eb, 1)
        
        with tf.name_scope('Attention_layer'):
            attention_output = self.din_attention(item_eb, item_his_eb, self.ATTENTION_SIZE, mask)
            att_fea = tf.reduce_sum(attention_output, 1)
            #tf.summary.histogram('att_fea', att_fea)
        inp = tf.concat([uid_batch_embedded, item_eb, item_his_eb_sum, item_eb * item_his_eb_sum, att_fea], -1)
        # Fully connected layer
        output = tf.squeeze(self.build_fcn_net(inp))
        return output
