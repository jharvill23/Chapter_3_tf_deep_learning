import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Generate synthetic data
N = 100
#Zeros form a Gaussian centered at (-1, -1)
#epsilon is 0.1

# <editor-fold desc="generate data">
x_zeros = np.random.multivariate_normal(
    mean=np.array((-1, -1)), cov=0.1*np.eye(2), size=(N//2,))
y_zeros = np.zeros((N//2,))
#Ones form a Gaussian centered at (1, 1)
#epsilon is 0.1
x_ones = np.random.multivariate_normal(
    mean=np.array((1, 1)), cov=0.1*np.eye(2), size=(N//2,))
y_ones = np.ones((N//2,))
x_np = np.vstack([x_zeros, x_ones])
y_np = np.concatenate([y_zeros, y_ones])
# </editor-fold>

# Generate tensorflow graph
with tf.name_scope('placeholders'):
    x = tf.placeholder(tf.float32, (N, 1))
    y = tf.placeholder(tf.float32, (N,))
with tf.name_scope('weights'):
    # Note that x is a scalar, so W is a single learnable weight
    W = tf.Variable(tf.random_normal((1, 1)))
    b = tf.Variable(tf.random_normal((1,)))
with tf.name_scope('prediction'):
    y_pred = tf.matmul(x, W) + b
with tf. name_scope('loss'):
    l = tf.reduce_sum((y - y_pred)**2)
# Add training op
with tf.name_scope('optim'):
    # Set learning rate to .001
    train_op = tf.train.AdamOptimizer(.001).minimize(l)
with tf.name_scope('summaries'):
    x = 5
# added some comments for github
dummy_var = 5
learning_rate = .001
W = tf.Variable((3,))
l = tf.reduce_sum(W)
gradW = tf.gradients(l, W)



n_steps = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Train model
    for i in range(n_steps):
        feed_dict = {x: x_np, y: y_np}
        _, summary, loss = sess.run([train_op])

# plt.scatter(x_np[:, 0], x_np[:, 1], c=y_np)
# plt.show()





stop = 5