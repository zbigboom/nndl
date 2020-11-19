import tensorflow as tf
# VGG-16
net=tf.keras.Sequential()
# conv1
net.add(tf.keras.layers.Conv2D(filters=64,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# conv2
net.add(tf.keras.layers.Conv2D(filters=64,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# pool1
net.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
# conv3
net.add(tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# conv4
net.add(tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# pool2
net.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
# conv5
net.add(tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# conv6
net.add(tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# conv7
net.add(tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# pool3
net.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
# conv8
net.add(tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# conv9
net.add(tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# conv10
net.add(tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# pool4
net.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
# conv11
net.add(tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# conv12
net.add(tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# conv13
net.add(tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# pool5
net.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

