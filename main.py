import tensorflow as tf
import numpy as np

path = 'E:\python\mnist\mnist.pkl.gz'
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path)

batch_size = 100
learning_ratr = 0.01
learning_rate_decay = 0.99
max_steps = 30000


def hidden_layer(input_tensor, reguarizer, avg_class, resuse):
    with tf.compat.v1.variable_scope("c1-conv", reuse=resuse):
        conv1_weights = tf.compat.v1.get_variable('weight', [5, 5, 1, 32],
                                            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.compat.v1.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    with tf.name_scope("s2-max_pool", ):
        pooll = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    with tf.compat.v1.variable_scope("c3-conv", reuse=resuse):
        conv2_weights = tf.compat.v1.get_variable("weight", [5, 5, 32, 64],
                                            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.compat.v1.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pooll, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    with tf.name_scope("s4-max_pool", ):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        shape = pool2.get_shape().as_list()
        nodes = shape[1] * shape[2] * shape[3]
        reshaped = tf.reshape(pool2, [shape[0]], nodes)
    with tf.compat.v1.variable_scope("layer5-gull", reuse=resuse):
        Full_connection1_weigths = tf.compat.v1.get_variable("bise", [512], initializer=tf.compat.v1.truncated_normal_initializer(0.1))
        tf.compat.v1.add_to_collection("losses", reguarizer(Full_connection1_weigths))
        Full_connection1_biases=tf.compat.v1.get_variable("bias",[512],initializer=tf.compat.v1.constant_initializer(0.1))
        if avg_class==None:
            Full_1=tf.nn.relu(tf.matmul(reshaped,Full_connection1_weigths)+Full_connection1_biases)
        else:
            Full_1=tf.nn.relu(tf.matmul(reshaped,avg_class.average(Full_connection1_weigths)+avg_class.average(Full_connection1_biases)))
    with tf.compat.v1.variable_scope("layer6-full2",reuse=resuse):
        Full_connection2_weigths=tf.compat.v1.get_variable("weight",[512,10],initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        tf.compat.v1.add_to_collection("losses",reguarizer(Full_connection2_weigths))
        Full_connection2_biases=tf.compat.v1.get_variable("bias",[10],initializer=tf.compat.v1.truncated_normal_initializer(0.1))
        if avg_class==None:
            result=tf.matmul(Full_1,Full_connection2_weigths)+Full_connection2_biases
        else:
            result=tf.matmul(Full_1,avg_class.average(Full_connection2_weigths))+avg_class.average(Full_connection2_biases)
    return result

tf.compat.v1.disable_eager_execution()
x=tf.compat.v1.placeholder(tf.float32,[batch_size,28,28,1],name="x-input")
y_=tf.compat.v1.placeholder(tf.float32,[None,10],name="y-input")
regularizer=tf.keras.regularizers.l2(0.0001)
training_step=tf.compat.v1.Variable(0,trainable=False)
variable_averages=tf.train.ExponentialMovingAverage(0.99,training_step)
variable_averages_op=variable_averages.apply(tf.compat.v1.trainable_variables())
average_y=hidden_layer(x,regularizer,variable_averages,resuse=True)
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_,labels=tf.argmax(y_,1))
cross_entropy_mean=tf.reduce_mean(cross_entropy)
loss=cross_entropy_mean+tf.add_n(tf.compat.v1.get_collection('losses'))
learning_rate=tf.compat.v1.train.exponential_decay(learning_ratr,training_step,mnist.train.num_rxamples/batch_size,learning_rate_decay,staircase=True)
train_step=tf.compat.v1.train.GradientDescentOptimizer(learning_ratr).minimize(loss,globals_step=training_step)
with tf.control_dependencies([train_step,variable_averages_op]):
    train_op=tf.no_op(name='train')
crorent_predicition=tf.equal(tf.compat.v1.arg_max(average_y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(crorent_predicition,tf.float32))
with tf.compat.v1.Session() as sess:
    tf.compat.v1.global_variables_initializer().run()
    for i in range(max_steps):
        if i%1000==0:
            x_val,y_val=mnist.validation.next_batch(batch_size)
            reshaped_x2=np.reshape(x_val,(batch_size,28,28,1))
            validate_feed={x:reshaped_x2,y_:y_val}
            validate_accuracy=sess.run(accuracy,feed_dict=validate_feed)
            print("After %d training step(s),validation accuracy""using average model is %g%%"%(i,validate_accuracy*100))
            x_train,y_train=mnist.train.next_batch(batch_size)
            reshaped_xs=np.reshape(x_train,(batch_size,28,28,1))
            sess.run(train_op,feed_dict={x:reshaped_xs,y_:y_train})