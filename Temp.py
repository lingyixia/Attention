# coding:utf-8

# -------------------------------------------------------------------------------
# @Author        chenfeiyu01
# @Name:         Temp.py
# @Project       Attention
# @Product       PyCharm
# @DateTime:     2019-07-19 11:19
# @Contact       chenfeiyu01@baidu.com
# @Version       1.0
# @Description:  
# -------------------------------------------------------------------------------
import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

tf.enable_eager_execution()
arg1 = tf.math.rsqrt(tf.constant(value=1))
print(arg1)