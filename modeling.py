import tensorflow as tf
from global_variables import hidden_units

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
  
