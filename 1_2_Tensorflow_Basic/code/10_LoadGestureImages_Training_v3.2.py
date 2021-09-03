# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 21:06:40 2019

@author: Yoo-Joo Choi
"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

#AUTOTUNE = tf.data.experimental.AUTOTUNE

import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

tf.__version__

import pathlib
import pdb
#data_dir = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
#                                         fname='flower_photos', untar=True)
data_dir = pathlib.Path('gesture_data', '10_dataset/mhi_2')
test_data_dir = pathlib.Path('gesture_data', '10_dataset/mhi_test')

image_count = len(list(data_dir.glob('*/*.png')))
image_count


test_data_count = len(list(test_data_dir.glob('*/*.jpg')))
test_data_count
print("test_data_count = %d" %(test_data_count))



CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
CLASS_NAMES


'''  Load using tf.keras.preprocessing
'''
# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 30
IMG_HEIGHT = 240
IMG_WIDTH = 320
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))
test_data_gen = image_generator.flow_from_directory(directory=str(test_data_dir),
                                                     batch_size=test_data_count, # There are 36 test images
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))


  
"""**[CNN-03]** 첫 번째 단계의 합성곱 필터와 풀링 계층을 정의한다."""

num_filters1 = 32

x = tf.placeholder(tf.float32, [None, 240, 320, 3], name="x")     
x_image = tf.reshape(x, [-1,240,320,3])

W_conv1 = tf.Variable(tf.truncated_normal([5,5,3,num_filters1], stddev=0.1))
h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME')

b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]))
h_conv1_cutoff = tf.nn.relu(h_conv1 - b_conv1) 

h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

"""**[CNN-04]** 두 번째 단계의 합성곱 필터와 풀링 계층을 정의한다."""

num_filters2 = 32

W_conv2 = tf.Variable( tf.truncated_normal([5,5,num_filters1,num_filters2], stddev=0.1))
h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME')

b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))
h_conv2_cutoff = tf.nn.relu(h_conv2 - b_conv2)

h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

"""**[CNN-05]** 전 결합층, 드롭아웃 계층, 소프트맥스 함수를 정의한다."""
h_pool2_flat = tf.reshape(h_pool2, [-1, 60*80*num_filters2])

num_units1 = 60*80*num_filters2
num_units2 = 32

w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))
hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat, w2) + b2)

keep_prob = tf.placeholder(tf.float32, name="keep_prob")
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

w0 = tf.Variable(tf.zeros([num_units2, 3]))
b0 = tf.Variable(tf.zeros([3]))
#p = tf.nn.softmax(tf.matmul(hidden2_drop, w0) + b0)
f = tf.matmul(hidden2_drop, w0) + b0
p = tf.nn.softmax(f)
p = tf.identity(p, "p")

"""**[CNN-06]** 오차 함수 loss, 트레이닝 알고리즘 train_step, 정답률 accuracy을 정의한다."""
t = tf.placeholder(tf.float32, [None, 3])
#loss = -tf.reduce_sum(t * tf.log(p))
#loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=p, labels=tf.stop_gradient(t)))
loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=f, labels=tf.stop_gradient(t)))
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""**[CNN-07]** 세션을 준비하고 Variable을 초기화한다."""
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

"""**[CNN-08]** 파라미터 최적화를  반복한다.
"""

batch_size = BATCH_SIZE
num_epochs = 100
num_iterations = int(image_count / batch_size)
print("num_iterations = %d" %(num_iterations))

epoch = 0
i = 0
for _ in range(num_epochs):
    epoch += 1
    avg_loss = 0 
    avg_accuracy = 0
    
    for iteration in range (num_iterations):
      i += 1
      #batch_xs, batch_ts = mnist.train.next_batch(batch_size)
      image_batch, label_batch = next(train_data_gen)

      #show_batch(image_batch, label_batch)
      _, loss_val, acc_val = sess.run([train_step, loss, accuracy], feed_dict={x: image_batch, t: label_batch, keep_prob:0.8})
      avg_loss += loss_val / num_iterations
      avg_accuracy += acc_val / num_iterations
      
      if i % 3 == 0:
          test_image_batch, test_label_batch = next(test_data_gen)
          loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x: test_image_batch, t: test_label_batch, keep_prob:1.0})
          
          #show_batch(test_image_batch, test_label_batch)
          print ('==> Step: %d, (Test Data) Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
          saver.save(sess, 'data/cnn_session', global_step=i)
      
    print("Epoch  :  %04d, (train Data) Loss: %0.9f,  Accuracy: %0.9f" %(epoch, avg_loss, avg_accuracy))
    
print("Learning Finished")
  
#Test model and check accuracy
test_image_batch, test_label_batch = next(test_data_gen)
print( " Accuracy for Test Data:", sess.run(accuracy, feed_dict={x:test_image_batch, t:test_label_batch, keep_prob:1.0}))



"""**[CNN-09]** 세션 정보를 저장한 파일이 생성되고 있음을 확인한다."""
  
