import tensorflow as tf

# VGG-16

net = tf.keras.Sequential()
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
net.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
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
net.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# conv5
net.add(tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# conv6
net.add(tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# conv7
net.add(tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# pool3
net.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# conv8
net.add(tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# conv9
net.add(tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# conv10
net.add(tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# pool4
net.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# conv11
net.add(tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# conv12
net.add(tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# conv13
net.add(tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               padding='same'))
# pool5
net.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# flatten
net.add(tf.keras.layers.Flatten())

# FC1
net.add(tf.keras.layers.Dense(4096, activation='relu'))
net.add(tf.keras.layers.Dropout(0.5))
# FC2
net.add(tf.keras.layers.Dense(4096, activation='relu'))
net.add(tf.keras.layers.Dropout(0.5))
# FC3 softmax
net.add(tf.keras.layers.Dense(1000, activation='softmax'))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.9, momentum=0.0, nesterov=False)

net.compile(loss='sparse_categorical_crossentory',
            optimizer=optimizer,
            metrics=['accuracy'])
