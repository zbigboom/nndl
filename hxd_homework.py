import tensorflow as tf
import os
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Dense, Dropout, Flatten

# 屏蔽通知信息和警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def mnist_dataset():
    # 下载mnist数据集
    (x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # 定义数据集大小28*28*1
    x = x.reshape(x.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # 格式化装载数据集
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(prepare_mnist_features_and_labels)
    test_ds = test_ds.take(20000).shuffle(20000).batch(2000)
    return ds, test_ds


def prepare_mnist_features_and_labels(x, y):
    # 将tensor转换为新的格式
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y


class myConvModel(tf.keras.Model):
    def __init__(self):
        super(myConvModel, self).__init__()
        # 第一个卷积层5*5*64步长为1
        self.conv1 = Conv2D(filters=64,
                            kernel_size=3,
                            strides=1,
                            activation='relu',
                            padding='same')
        # 第一个汇聚层最大池化3*3步长为2
        self.pool1 = MaxPool2D(pool_size=3, strides=2, padding='same')
        # 第二个卷积层3*3*64步长为1
        self.conv2 = Conv2D(filters=64,
                            kernel_size=3,
                            strides=1,
                            activation='relu',
                            padding='same')
        # 第二个汇聚层最大池化3*3步长为2
        self.pool2 = MaxPool2D(pool_size=3, strides=2, padding='same')
        # 第三个卷积层3*3*128步长为1
        self.conv3 = Conv2D(filters=128,
                            kernel_size=3,
                            strides=1,
                            activation='relu',
                            padding='same')
        # 第四个卷积层3*3*128步长为1
        self.conv4 = Conv2D(filters=128,
                            kernel_size=3,
                            strides=1,
                            activation='relu',
                            padding='same')
        # 第五个卷积层3*3*256步长为1
        self.conv5 = Conv2D(filters=256,
                            kernel_size=3,
                            strides=1,
                            activation='relu',
                            padding='same')
        # 第三个汇聚层最大池化3*3步长为2
        self.pool3 = MaxPool2D(pool_size=3, strides=2, padding='same')
        # 数据拉直
        self.flat = Flatten()
        # 第一个全连接层
        self.dense1 = Dense(1024, activation='relu')
        # 防止过拟合
        self.dro1=Dropout(0.5)
        # 第二个全连接层
        self.dense2 = Dense(1024, activation='relu')
        # 防止过拟合
        self.dro2=Dropout(0.5)
        # 第三个全连接层
        self.dense3 = Dense(10, activation='softmax')

    @tf.function
    # 将模型转换为易于部署且高性能的 TensorFlow 图模型
    def call(self, x):
        # 输入输出连接
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        pool3 = self.pool3(conv5)
        flat = self.flat(pool3)
        dense1 = self.dense1(flat)
        dro1=self.dro1(dense1)
        dense2 = self.dense2(dro1)
        dro2=self.dro2(dense2)
        logits = self.dense3(dro2)
        return logits

# 模型建立
model = myConvModel()
# 优化器
optimizer = tf.optimizers.Adam()


@tf.function
# 计算损失值
def compute_loss(logits, labels):
    # 计算跨张量维度的元素的均值
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))
# 交叉熵函数输入前面定义为logits为float32，labels为int64

@tf.function
# 计算准确度
def compute_accuracy(logits, labels):
    # 返回跨张量轴的最大值的索引
    predictions = tf.argmax(logits, axis=1)
    # 以特定格式返回计算跨张量维度的元素的均值
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


@tf.function
# 训练一次
def train_one_step(model, optimizer, x, y):
    # 记录操作以自动区分
    with tf.GradientTape() as tape:
        # logits代表计算softmax交叉熵后反向传播的梯度
        logits = model(x)
        # y值为标签
        loss = compute_loss(logits, y)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        accuracy = compute_accuracy(logits, y)

        return loss, accuracy

# 测试一次
def test_step(model, x, y):
    logits = model(x)
    loss = compute_loss(logits, y)
    accuracy = compute_accuracy(logits, y)
    return loss, accuracy


def train(epoch, model, optimizer, ds):
    loss = 0.0
    accuracy = 0.0
    for step, (x, y) in enumerate(ds):
        loss, accuracy = train_one_step(model, optimizer, x, y)
        if step % 500 == 0:
            print('epoch', epoch, ':loss', loss.numpy(), ':accuracy', accuracy.numpy())
    return loss, accuracy


def test(model, ds):
    loss = 0.0
    accuracy = 0.0
    for step, (x, y) in enumerate(ds):
        loss, accuracy = test_step(model, x, y)

    print('test loss', loss.numpy(), ':accuracy', accuracy.numpy())
    return loss, accuracy


train_ds, test_ds = mnist_dataset()
for epoch in range(2):
    loss, accuracy = train(epoch, model, optimizer, train_ds)
loss, accuracy = test(model, test_ds)
