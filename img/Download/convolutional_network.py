from __future__ import print_function

import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from StringIO import StringIO
import time

# Parameters
model_file = 'model.ckpt'

learning_rate = 0.0001
training_iters = 10000
batch_size = 2 #
display_step = 1

NUM_CLASSES = 2
IMAGE_ROW = 600
IMAGE_COL = 300
CHANNELS = 1
IMAGE_PIXELS = IMAGE_ROW * IMAGE_COL * CHANNELS

CONV1_DEPTH = 16
CONV2_DEPTH = 32
# Network Parameters
n_input = IMAGE_PIXELS
n_classes = 2 # First try 2 classes
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
    # 5x5 conv, 1 input, 16 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, CONV1_DEPTH])),
    # 5x5 conv, 16 inputs, 32 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, CONV1_DEPTH, CONV2_DEPTH])),
    # fully connected, 150*75*64 inputs, 1024 outputs, 300/4 * 600/4 * 64
    'wd1': tf.Variable(tf.random_normal([150*75*CONV2_DEPTH, 512])),
    # 512 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([512, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bd1': tf.Variable(tf.random_normal([512])),
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

# Save model
saver = tf.train.Saver()
# Initializing the variables
init = tf.initialize_all_variables()

def getImage(url):
    return Image.open(StringIO(requests.get(url).content)).convert('L')
def loadData(filename):
    print ("Loading images from " + filename)
    total = 0
    train_images = []
    pos = open(filename, "r")
    for line in pos:
        url, label = line.strip().split('\t')
        image = getImage(url)
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
        total += 1
    ti = np.array(train_images)
    '''
    print (ti.shape)
    print (total)
    print (IMAGE_PIXELS)
    '''
    X = ti.reshape(total, IMAGE_PIXELS)
    return X
X_pos = loadData("pos.train.txt")
X_neg = loadData("neg.train.txt")

batch_size = X_pos.shape[0] + X_neg.shape[0]
prev_acc = 0.0

print ("Number of positive examples: " + str(X_pos.shape[0]))
print ("Number of negative examples: " + str(X_neg.shape[0]))

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        n_pos = X_pos.shape[0]
        n_neg = X_neg.shape[0]
        #Use whole data
        batch_x = np.concatenate((X_pos, X_neg), axis=0)

        a = np.zeros(n_pos, dtype=np.int)
        b = np.zeros((n_pos, 2))
        b[np.arange(n_pos), a] = 1
        c = np.ones(n_neg, dtype=np.int)
        d = np.zeros((n_neg, 2))
        d[np.arange(n_neg), c] = 1
        
        batch_y = np.concatenate((b, d), axis=0)

        start_time = time.time()
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        duration = time.time() - start_time
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_x.shape[0]) + ", batch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc) + ", Time(sec)= " + "{:.3f}".format(duration))

        X_pos_dev = loadData("pos.dev.txt")
        X_neg_dev = loadData("neg.dev.txt")
        batch_x_dev = np.concatenate((X_pos_dev, X_neg_dev), axis=0)
        n_pos = X_pos_dev.shape[0]
        n_neg = X_neg_dev.shape[0]
        a = np.zeros(n_pos, dtype=np.int)
        b = np.zeros((n_pos, 2))
        b[np.arange(n_pos), a] = 1
        c = np.ones(n_neg, dtype=np.int)
        d = np.zeros((n_neg, 2))
        d[np.arange(n_neg), c] = 1
        
        batch_y_dev = np.concatenate((b, d), axis=0)
        print("Dev" + str(X_pos.shape[0]))
        dev_acc = sess.run(accuracy, feed_dict={x: batch_x_dev,
                                      y: batch_y_dev,
                                      keep_prob: 1.})
        print("Dev Data Accuracy:", dev_acc)
        if dev_acc > prev_acc:
            # Save model
            saver.save(sess, model_file)
            prev_acc = dev_acc
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for test data
    X_pos = loadData("pos.test.txt")
    X_neg = loadData("neg.test.txt")
    batch_x = np.concatenate((X_pos, X_neg), axis=0)
    n_pos = X_pos.shape[0]
    n_neg = X_neg.shape[0]
    a = np.zeros(n_pos, dtype=np.int)
    b = np.zeros((n_pos, 2))
    b[np.arange(n_pos), a] = 1
    c = np.ones(n_neg, dtype=np.int)
    d = np.zeros((n_neg, 2))
    d[np.arange(n_neg), c] = 1
        
    batch_y = np.concatenate((b, d), axis=0)
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: batch_x,
                                      y: batch_y,
                                      keep_prob: 1.}))
