import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

BATCH_SIZE = 100
LEARNING_RATE_BASE =  0.005
LEARNING_RATE_DECAY = 0.99
STEPS = 30000
REGULARIZER = 0.0001
# data
mnist = input_data.read_data_sets("../mnist/", one_hot=True)


def weight_variable(shape, name, regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1,name=name))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

# define placeholder
with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 10], name='y_input')

# conv1 layer
with tf.name_scope('layer_conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1', regularizer=0.0001)
    b_conv1 = bias_variable([32], name='b_conv1')
    h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1, name='h_conv1')

#  pool1 layer
with tf.name_scope('layer_pool1'):
    h_pool1 = max_pool_2x2(h_conv1, name='h_pool1')

# conv2 layer
with tf.name_scope('layer_conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2', regularizer=0.0001)
    b_conv2 = bias_variable([64], name='b_conv2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name='h_conv2')

# pool2 layer
with tf.name_scope('layer_pool2'):
    h_pool2 = max_pool_2x2(h_conv2, name='h_pool2')

# func1 layer
with tf.name_scope('layer_func1'):
    W_func1 = weight_variable([7 * 7 * 64, 1024], name='W_func1', regularizer=0.0001)
    b_func1 = bias_variable([1024], name='b_func1')
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name='h_pool2_flat')
    h_func1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_func1) + b_func1, name='h_func1')


# output layer
with tf.name_scope('softmax'):
    W_func2 = weight_variable([1024, 10], name='W_func2', regularizer=0.0001)
    b_func2 = bias_variable([10], name='b_func2')
    prediction = tf.nn.softmax(tf.matmul(h_func1, W_func2) + b_func2, name='prediction')
    #prediction = tf.matmul(h_func1,W_func2) + b_func2
# cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
tf.summary.scalar('loss',loss)

global_step = tf.Variable(0,trainable=False)
learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,   #学习率多久更新一次
        LEARNING_RATE_DECAY,            #学习率衰减率
        staircase=True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

# accuracy
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy',accuracy)

# initial session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# tensorboard
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('../logs', sess.graph)

# # save model
saver = tf.train.Saver()

output_node_names = 'input/x_input,softmax/prediction'
# train

#for step in range(mnist.train.images.shape[0]):
for i in range(STEPS):
    batch = mnist.train.next_batch(100)
    batch_xs = batch[0].reshape([100, 28, 28, 1])
    batch_ys = batch[1]

    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch[1]})
    if i % 100 == 0:
        loss_value, acc, step = sess.run((loss, accuracy,global_step), feed_dict={xs:batch_xs, ys:batch_ys})
        print('Current step: {}, loss: {}, accuracy: {:.2%}'.format(step, loss_value, acc))
        rs = sess.run(merged,feed_dict={xs:batch_xs,ys:batch_ys})
train_writer.add_summary(rs,step)
# 第一种方式，保存为ckpt文件
# saver.save(sess, "trained_model/mnist_cnn.ckpt", global_step=step)

# 第二种方式，保存为pb文件
constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names.split(','))
with tf.gfile.GFile('../model/model.pb', "wb") as f:
    f.write(constant_graph.SerializeToString())


sess.close()
