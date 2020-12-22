import tensorflow as tf


# 局部响应归一化LRN
class LRN(tf.keras.layers.Layer):
    def __init__(self):
        super(LRN, self).__init__()
        self.depth_radius = 2
        self.bias = 1
        self.alpha = 1e-4
        self.beta = 0.75

    def call(self, x):
        return tf.nn.lrn(x, depth_radius=self.depth_radius,
                         bias=self.bias,
                         alpha=self.alpha,
                         beta=self.beta)


# 数据集为224*224*3的图像输出为1000个类
# imagenet数据集
net = tf.keras.models.Sequential()
# 添加第一个卷积层大小为11*11*3*48步长为4零填充3，两个卷积核，得到两个55*55*48的特征映射组
net.add(tf.keras.layers.Conv2D(filters=96,
                               kernel_size=11,
                               strides=4,
                               activation='relu',
                               padding='same'))
# 第一个汇聚层，最大池化3*3大小步长为2，得到两个27*27*48的特征映射组
net.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
# 归一化操作
net.add(LRN())
# 第二个卷积层大小为5*5*48*128步长为1 零填充2，两个卷积核，得到两个27*27*128的特征映射组
net.add(tf.keras.layers.Conv2D(filters=256,
                               kernel_size=5,
                               strides=1,
                               activation='relu',
                               padding='same'))
# 第二个汇聚层，最大池化3*3大小步长为2，得到两个13*13*128
net.add(tf.keras.layers.MaxPooling2D(pool_size=3, strides=2))
# 归一化操作
net.add(LRN())
# 第三个卷积层两个路径的融合，用一个3*3*256*384的卷积核步长为1零填充为1，得到两个13*13*192的特征映射组
net.add(tf.keras.layers.Conv2D(filters=384,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# 第四个卷积层用两个卷积核3*3*192*192步长为2，得到两个大小为13*13*192的特征映射组
net.add(tf.keras.layers.Conv2D(filters=384,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# 第五个卷积层用两个卷积核3*3*192*128步长为1，得到两个大小为13*13*128的特征映射组
net.add(tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# 第三个汇聚层，最大池化3*3*192*128步长为1，得到两个6*6*128的特征映射组
net.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
# 三个全链接层
# 将数据拉直
net.add(tf.keras.layers.Flatten())
# 第一个全链接层激活函数为relu
net.add(tf.keras.layers.Dense(4096, activation='relu'))
# 防止过拟合，Dropout是使得一定的概率暂时从网络中丢弃，简单来说就是在前向传播时让某个神经元的激活值以一定的概率p停止工作
net.add(tf.keras.layers.Dropout(0.5))
# 第二个全链接层激活函数为relu
net.add(tf.keras.layers.Dense(4096, activation='relu'))
# 防止过拟合
net.add(tf.keras.layers.Dropout(0.5))
# 第三个全链接层激活函数为softmax
net.add(tf.keras.layers.Dense(1000, activation='softmax'))
# 建立网络，损失函数和训练算法采用交叉熵函数(cross entropy)和批量随机梯度下降（SGD）
optimizer = tf.keras.optimizers.SGD(learning_rate=0.9, momentum=0.0, nesterov=False)
net.compile(loss='sparse_categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
