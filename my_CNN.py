# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import numpy as np 
import matplotlib as mp
mp.use('Agg')
#%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import math

FLAGS = None

#-----------------------HYPERPARAMETERS-----------------------

learning_rate = 1e-3
batch_size = 100
training_iteration = 50

#-----------------------HYPERPARAMETERS-----------------------
def id(array):
    id = 0
    for i in range(0,5):
        for j in range(0,5):
            id += array[i][j]
    return id


def deepnn(x):

  x_images = tf.reshape(x, [-1 , 28 , 28 ,1])

  #conv1

  W_conv1 = weight_variable([5,5,1,32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_images,W_conv1) + b_conv1)



  #pool1

  h_pool1 = max_pool_2x2(h_conv1)


  #conv2

  W_conv2 = weight_variable([5,5,32,64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)

  #pool2

  h_pool2 = max_pool_2x2(h_conv2)


  #fc1

  W_fc1 = weight_variable([7*7*64 , 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2 , [-1 , 7*7*64])

  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat , W_fc1) + b_fc1)

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  return y_conv, keep_prob , W_conv1 , W_conv2 , h_conv1 , h_conv2


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


image_id_file = open('image_id_file.txt' , 'w')


def main(_):

  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  x = tf.placeholder(tf.float32 , [None , 784])

  y_ = tf.placeholder(tf.float32 , [None , 10])

  y_conv , keep_prob , W_conv1 , W_conv2, h_conv1 , h_conv2 = deepnn(x)


  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_ , logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)


  correct_prediction = tf.equal(tf.argmax(y_,1) , tf.argmax(y_conv,1) )
  correct_prediction = tf.cast(correct_prediction , tf.float32)

  accuracy = tf.reduce_mean(correct_prediction)

  train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


  ####################################################################
  W_conv1_filters = []
  for i in range(0,32):
    conv1_filter = tf.reshape(W_conv1[:,:,:,i] , [5,5])
    W_conv1_filters.append(conv1_filter)

  
  W_conv2_filters = []
  for v in range(0,2):
    for c in range(0,64):
      conv2_filter = tf.reshape(W_conv2[:,:,v,c] , [5,5])
      W_conv2_filters.append(conv2_filter)
    
    W_conv2_filters.append(tf.reshape([[0,0,0,0,0],[0,0,50,0,0],[0,0,0,0,0],[0,0,20,0,0],[0,80,0,0,0]] , [5,5]))
  ####################################################################

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(training_iteration):
      batch = mnist.train.next_batch(batch_size)

      train_step.run(feed_dict={x: batch[0] , y_:batch[1] , keep_prob: 0.5})

    print("Accuracy: %g" % accuracy.eval(feed_dict={x: mnist.test.images , y_: mnist.test.labels , keep_prob: 1.0}))
    k = 1
    for W_conv1_filter  in W_conv1_filters :
      image_array1 = sess.run(W_conv1_filter)
      
      plt.figure(1 , figsize=(10,10))
      plt.subplot(6,6,k)
      plt.title(' ')
    	
      #interpolations = [None , "nearest" ,"bilinear" , "bicubic" , "gaussian"]
    	
      plt.imshow(image_array1, interpolation="none", cmap="gray" )
      k += 1

    plt.show()
    plt.savefig('First_Layer_Filters')

    m = 1

    for W_conv2_filter  in W_conv2_filters :
      image_array2 = sess.run(W_conv2_filter)
      
      plt.figure(1 , figsize=(10,10))
      plt.subplot(17,8,m)
      image_id = str(id(image_array2))
      image_id_file.write(image_id)
      image_id_file.write('\n')
      
      plt.title(' ')
    	
      #interpolations = [None , "nearest" ,"bilinear" , "bicubic" , "gaussian"]
    	
      plt.imshow(image_array2, interpolation="none", cmap="gray" )
      m += 1

    plt.show()
    plt.savefig('Second_Layer_Filters')

    image_to_visualize_1 = np.reshape(mnist.test.images[7] , (1,784)) 

    sample_image = sess.run(h_conv1, feed_dict={x: image_to_visualize_1})

    print(sample_image.shape)

    

    sample_images = np.reshape(sample_image[0,:,:,0] ,(28,28) )
    
    print(sample_image.shape)
    
    for z in range(1,13):
        plt.figure(1 , figsize=(10,10))
        plt.subplot(4,3,z)
        plt.imshow(sample_image[0,:,:,z], interpolation="none", cmap="gray" )
    plt.show()

    plt.savefig('sample_image_0') 

    #---------------------------------------------------------------
     
    image_to_visualize_2 = np.reshape(mnist.test.images[7] , (1,784)) 

    sample_image_2 = sess.run(h_conv2, feed_dict={x: image_to_visualize_2})

    print(sample_image_2.shape)

    

    #sample_image_2 = np.reshape(sample_image_2[0,:,:,:] ,(14,14,64) )
    
    print(sample_image_2.shape)
    
    for t in range(1,13):
        plt.figure(1 , figsize=(10,10))
        plt.subplot(4,3,t)
        plt.imshow(sample_image_2[0,:,:,t], interpolation="none", cmap="gray" )
    plt.show()

    plt.savefig('sample_image_2')



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
