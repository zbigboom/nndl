import tensorflow as tf



class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filter_num, kernel_size=(3, 3), strides=stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')

        self.conv2 = tf.keras.layers.Conv2D(filter_num, kernel_size=(3, 3), strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filter_num, kernel_size=(1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)

        output = tf.keras.layers.add([out, identity])
        output = tf.nn.relu(output)

        return output


class Net(tf.keras.Model()):
    def __init__(self, layer_dims, num_classes=10):
        super(Net, self).__init__()
        self.stem = tf.keras.Sequential([tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                         tf.keras.layers.BatchNormalization(),
                                         tf.keras.layers.Activation('relu'),
                                         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                         ])

        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()

        self.fc = tf.keras.layers.Dense(num_classes)

    def __call__(self, inputs, training=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = self.fc(x)

        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = tf.keras.Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks
