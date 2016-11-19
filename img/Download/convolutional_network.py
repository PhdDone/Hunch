from __future__ import print_function

import tensorflow as tf
from PIL import Image
import numpy as np

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 15
batch_size = 2 #128
display_step = 1

NUM_CLASSES = 2
IMAGE_ROW = 600
IMAGE_COL = 300
CHANNELS = 1
IMAGE_PIXELS = IMAGE_ROW * IMAGE_COL * CHANNELS

# Network Parameters
n_input = IMAGE_PIXELS
n_classes = 2 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    print (x)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, IMAGE_ROW, IMAGE_COL, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 150*75*64 inputs, 1024 outputs, 300/4 * 600/4 * 64
    'wd1': tf.Variable(tf.random_normal([150*75*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        train_images = []
        files = ['02.jpg', '03.jpg']
        for filename in files:
            image = Image.open(filename).convert('L')
            width, height = image.size
            new_width = IMAGE_COL
            new_height = IMAGE_ROW
            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2
            image = image.crop((left, top, right, bottom))
            image = image.resize((IMAGE_COL,IMAGE_ROW))
            train_images.append(np.array(image))

        ti = np.array(train_images)
        batch_x = ti.reshape(len(files), IMAGE_PIXELS)
        label = [[1.0,0.0], [0.0,1.0]]
        
        batch_y = np.array(label)
        print(len(batch_x[0]))
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        #print (len(batch_x[0]))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    # Save model??
    
    # Calculate accuracy for 256 mnist test images
    train_images = []
    files = ['02.jpg', '03.jpg']
    for filename in files:
        image = Image.open(filename).convert('L')
        width, height = image.size
        new_width = IMAGE_COL
        new_height = IMAGE_ROW
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        image = image.crop((left, top, right, bottom))
        image = image.resize((IMAGE_COL,IMAGE_ROW))
        train_images.append(np.array(image))
    ti = np.array(train_images)
    batch_x = ti.reshape(2, IMAGE_PIXELS)
    label = [[1.0,0.0], [0.0,1.0]]
        
    batch_y = np.array(label)
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: batch_x,
                                      y: batch_y,
                                      keep_prob: 1.}))
