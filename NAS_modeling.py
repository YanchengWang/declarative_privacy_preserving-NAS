import tensorflow as tf
from global_variables import hidden_units
import numpy as np

class MySequentialModel(tf.keras.Model):
    def __init__(self, out_size: int, **kwargs):
        super().__init__(**kwargs)
        self.dense_1 = tf.keras.layers.Dense(units=hidden_units, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(units=out_size, activation='softmax')

    def call(self, x):
        # return self.dense_2(x)
        return self.dense_2(self.dense_1(x))

class CrossProductOutputModel(tf.keras.Model):
    def __init__(self, out_size: int, **kwargs):
        super().__init__(**kwargs)
        self.dense_1 = tf.keras.layers.Dense(units=hidden_units, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(units=out_size, activation='sigmoid')

    def call(self, x):
        return self.dense_2(self.dense_1(x))

def gumbel_softmax(logits, tau, axis=-1):
    shape = tf.keras.backend.int_shape(logits)
    
    # Gumbel(0, 1)
    if len(shape) == 1:
        gumbels = tf.math.log(tf.random.gamma(shape, 1))
    else:
        gumbels = tf.math.log(
            tf.random.gamma(shape[:-1], [1 for _ in range(shape[-1])])
        )
        
    # Gumbel(logits, tau)
    gumbels = (logits + gumbels) / tau
    
    y_soft = tf.nn.softmax(gumbels, axis=axis)
    
    return y_soft

class superNet(tf.keras.Model):
    def __init__(self, channel_choices, flop_choices, out_size: int, **kwargs):
        super().__init__(**kwargs)
        
        self.channel_choices = channel_choices
        self.flop_choices = flop_choices

        self.max_channel = max(channel_choices)
        num_choices = len(dense_1_choices_logit)
        dense_1_choices_logit = np.ones((num_choices))/num_choices
        self.dense_1_choices_logit = tf.Variable(dense_1_choices_logit, tf.float64)

        self.dense_1_choices=[]
        for channel_num in channel_choices:
            self.dense_1_choices.append(tf.keras.layers.Dense(units=channel_num, activation='relu'))

        self.dense_2 = tf.keras.layers.Dense(units=out_size, activation='sigmoid')
    
    def call(self, tau, x):
        h_layer_1 = []

        for channel_num, dense_1 in zip(self.channel_choices, self.dense_1_choices):
            h = dense_1(x)
            paddings = tf.constant([[0, 0,], [0, self.max_channel-channel_num]])
            h = tf.pad(h, paddings, mode='CONSTANT', constant_values=0)
            h_layer_1.append(h)

        y = tf.stack(h_layer_1, axis=0)
        weights = gumbel_softmax(self.dense_1_choices_logit, tau)
        flops = tf.multiply(weights, self.flop_choices)
        y = tf.multiply(weights, y)
        y = tf.math.reduce_sum(y, axis=0, keepdims=False)

        y = self.dense_2(y)
        return y, flops