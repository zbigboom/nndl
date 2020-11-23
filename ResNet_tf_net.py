import tensorflow as tf

# 定义残差模块，初始化阶段创建残差模块中需要的卷积层激活函数层
class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = tf.keras.layers.Conv2D(filter_num, kernel_size=(3, 3), strides=stride, padding='same')
        # 规范化
        self.bn1 = tf.keras.layers.BatchNormalization()
        # 激活函数
        self.relu = tf.keras.layers.Activation('relu')
        # 第二个卷积层
        self.conv2 = tf.keras.layers.Conv2D(filter_num, kernel_size=(3, 3), strides=1, padding='same')
        # 规范化
        self.bn2 = tf.keras.layers.BatchNormalization()
        # 通过1*1的卷积完成shape匹配
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filter_num, kernel_size=(1, 1), strides=stride))
        # shape匹配完成直接连接
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        # 通过第一个卷积层
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        # 通过第二个卷积层
        out = self.conv2(out)
        out = self.bn2(out)
        # 通过identity模块
        identity = self.downsample(inputs)
        # 两条路径输出相加
        output = tf.keras.layers.add([out, identity])
        output = tf.nn.relu(output)

        return output

# 实现ResNet类
class Net(tf.keras.Model()):
    def __init__(self, layer_dims, num_classes=10):
        super(Net, self).__init__()
        # 预处理
        self.stem = tf.keras.Sequential([tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                         tf.keras.layers.BatchNormalization(),
                                         tf.keras.layers.Activation('relu'),
                                         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                         ])
        # 堆叠四个Block，每个包含多个BasicBlock设置步长不一样
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        # 通过pooling层将宽高降为1*1
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        # 全连接层
        self.fc = tf.keras.layers.Dense(num_classes)

    def __call__(self, inputs, training=None):
        # 通过预处理
        x = self.stem(inputs)
        # 通过四个模块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 通过池化层
        x = self.avgpool(x)
        # 通过全连接层
        x = self.fc(x)

        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        # 辅助函数,堆叠filter_num个BasicBlock
        res_blocks = tf.keras.Sequential()
        # 只有第一个BasicBlock步长可能不为1,实现下采样
        res_blocks.add(BasicBlock(filter_num, stride))
        # 剩余的步长都为1
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks
