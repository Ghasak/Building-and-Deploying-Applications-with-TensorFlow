import os
import tensorflow as tf

'''
Turn-off TensorFlow Warning messages in program output
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define computational graph

X = tf.compat.v1.placeholder(tf.float32, name = 'X')  # Placholder Node

Y = tf.compat.v1.placeholder(tf.float32, name = 'Y')

addtion = tf.add(X,Y, name = 'addition') # Add the nodes

# Create the session
with tf.compat.v1.Session() as session:
    result =  session.run(addtion, feed_dict = {X: [1, 2, 10], Y: [4, 2, 10]})

    print(result)
