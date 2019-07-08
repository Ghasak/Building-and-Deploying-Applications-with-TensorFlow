import os
import tensorflow as tf

'''
Turn-off TensorFlow Warning messages in program output
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define computational graph

X = tf.placeholder(tf.float32, name = 'X')  #Placholder Node

Y = tf.placeholder(tf.float32, name = 'Y')

addtion = tf.add(X,Y, name = 'addition') # Add the nodes

# Create the session
with tf.Session() as session:
    result =  session.run(addtion, feed_dict = {X: [1], Y: [4]})

    print(result)
