import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())

x = tf.Variable(-2.0)

with tf.GradientTape() as tape:
    y = x ** 2
    
df = tape.gradient(y, x)
print(df)