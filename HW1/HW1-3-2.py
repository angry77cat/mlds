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

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 50
display_epoch = 1
logs_path = '/home/master/05/john81923//MLDS2018/TensorBoard/'

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 784], name='InputData')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

# Set model weights
# W = tf.Variable(tf.zeros([784, 10]), name='Weights')
# b = tf.Variable(tf.zeros([10]), name='Bias')
train_loss=[]
test_loss=[]
train_acc=[]
test_acc=[]
param_num=[]
pa = [ 64, 128 , 256 ,384 ]
pb = [64, 128 , 256 ,384, 512 ]
for aa in pa:
    for bb in pb:
        dense2 = tf.layers.dense(inputs=x, units=aa , activation=tf.nn.softmax)
        dense3 = tf.layers.dense(inputs=dense2, units=bb ,activation=tf.nn.softmax)
        #dense4 = tf.layers.dense(inputs=dense3, units=256 , activation=tf.nn.softmax)
        #dense44 = tf.layers.dense(inputs=dense4, units=256 , activation=tf.nn.tanh)
        #dense444 = tf.layers.dense(inputs=dense44, units=256 , activation=tf.nn.tanh)
        dense5 = tf.layers.dense(inputs=dense3, units=10 ,activation=tf.nn.softmax)
        num_of_val =  np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print (" ")
        print (' param : ', num_of_val )
        
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
        tf.summary.scalar("param_loss", cost)
        # Create a summary to monitor accuracy tensor
        tf.summary.scalar("param_accuracy", acc)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()



        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            # op to write logs to Tensorboard
            summary_writer = tf.summary.FileWriter(logs_path+'param_train', graph=tf.get_default_graph())
            summary_writer2 = tf.summary.FileWriter(logs_path+'param_test', graph=tf.get_default_graph())

            avg_cost_old = 0.
            avg_test_cost_old = 0.
            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                avg_test_cost = 0.
                total_batch = int(mnist.train.num_examples/batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    t_batch_xs, t_batch_ys = mnist.test.next_batch(batch_size)
                    # Run optimization op (backprop), cost op (to get loss value)
                    # and summary nodes
                    _  = sess.run(optimizer ,feed_dict={x: batch_xs, y: batch_ys})
                    # Write logs at every iteration
                    
                    # Compute average loss
                    
                # Display logs per epoch step
                if (epoch+1) % display_epoch == 0:
                    for i in range(total_batch):
                        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                        t_batch_xs, t_batch_ys = mnist.test.next_batch(batch_size)
                        c = sess.run( cost ,
                                            feed_dict={x: batch_xs, y: batch_ys})
                        test_lossss  = sess.run( cost ,
                                                feed_dict={x: t_batch_xs, y: t_batch_ys})
                        avg_cost += c / total_batch
                        avg_test_cost += test_lossss / total_batch


                    print("Epoch:", '%04d' % (epoch+1), "  cost=", "{:.9f}".format(avg_cost))
                    print("     :", '%04d' % (epoch+1), "testcost=", "{:.9f}".format(avg_test_cost))

                    if  abs( avg_cost - avg_cost_old ) < 5e-3:
                        break
                    
                    
                    avg_cost_old = avg_cost
                    avg_test_cost_old = avg_test_cost
                    #summary_writer.add_summary(summary,  num_of_val )
                    #summary_writer2.add_summary(test_sum, num_of_val )
                    

            train_loss.append(avg_cost)
            test_loss.append(avg_test_cost)
            train_acc.append(acc.eval({x: mnist.train.images, y: mnist.train.labels}))
            test_acc.append(acc.eval({x: mnist.test.images, y: mnist.test.labels}))
            param_num.append(num_of_val)
            print("Optimization Finished!")

            # Test model
            # Calculate accuracy
            print("Accuracy:", acc.eval({x: mnist.test.images, y: mnist.test.labels}))
    plt.figure(0)
    plt.title('loss')
    plt.plot(  param_num,train_loss, 'ro', param_num,test_loss, 'bs')
    #plt.axis([-2, 2, 0, 5000000 ])
    plt.savefig(logs_path+'param_loss.jpg', bbox_inches='tight')   

    plt.figure(1)
    plt.title('acc')
    plt.plot(  param_num,train_acc, 'ro', param_num, test_acc,'bs')
    #plt.axis([-2, 2, 0, 5000000 ])
    plt.savefig(logs_path+'param_acc.jpg', bbox_inches='tight')   

    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")