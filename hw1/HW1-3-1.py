'''
Graph and Loss visualization using Tensorboard.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 4000
batch_size = 40
display_epoch = 5
logs_path = '/home/master/05/john81923//MLDS2018/TensorBoard/'

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 784], name='InputData')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

# Set model weights
W = tf.Variable(tf.zeros([784, 10]), name='Weights')
b = tf.Variable(tf.zeros([10]), name='Bias')

dense2 = tf.layers.dense(inputs=x, units=256 , activation=tf.nn.softmax)
dense3 = tf.layers.dense(inputs=dense2, units=256 ,activation=tf.nn.softmax)
dense4 = tf.layers.dense(inputs=dense3, units=256 , activation=tf.nn.softmax)
#dense44 = tf.layers.dense(inputs=dense4, units=256 , activation=tf.nn.tanh)
#dense444 = tf.layers.dense(inputs=dense44, units=256 , activation=tf.nn.tanh)
dense5 = tf.layers.dense(inputs=dense4, units=10 ,activation=tf.nn.softmax)


# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
    # Model
    #pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
    pred = dense5
with tf.name_scope('Loss'):
    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
with tf.name_scope('SGD'):
    # Gradient Descent
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
with tf.name_scope('Accuracy'):
    # Accuracy
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

indices = np.random.randint(10, size=(55000))
lab = []
for i in range (55000):
    lb = np.zeros(10)
    lb[ indices[i] ] += 1
    lab.append(lb)
lab = np.asarray(lab)
lab_bt = np.reshape(lab, [-1, batch_size ,10])

x_sh_lt = mnist.train.images
x_sh_lt_bt = np.reshape( x_sh_lt, [-1, batch_size, 784])



# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path +'train', graph=tf.get_default_graph())
    summary_writer2 = tf.summary.FileWriter(logs_path +'test', graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        avg_test_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = x_sh_lt_bt[ i ]
            batch_ys = lab_bt[ i ]
            t_batch_xs, t_batch_ys = mnist.test.next_batch(batch_size)
            # Run optimization op (backprop), cost op (to get loss value)
            # and summary nodes
            _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                     feed_dict={x: batch_xs, y: batch_ys})
            
            # Write logs at every iteration
            if i % 20 == 0:
              test_sum = sess.run( merged_summary_op ,
                                     feed_dict={x: t_batch_xs, y: t_batch_ys})
              summary_writer.add_summary(summary, epoch * total_batch + i)
              summary_writer2.add_summary(test_sum, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
            #avg_test_cost += test_loss / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1), "  cost=", "{:.9f}".format(avg_cost))
            #print("     :", '%04d' % (epoch+1), "testcost=", "{:.9f}".format(avg_test_cost))
            
            
    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    print("Accuracy:", acc.eval({x: mnist.test.images, y: mnist.test.labels}))

    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")