import tensorflow as tf


# 定义卷积操作
def conv2d(x,
           filters,
           num_row,
           num_col,
           padding='same',
           strides=(1, 1),
           name=None):
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=(num_row, num_col),
                               strides=strides,
                               padding=padding,
                               use_bias=False
                               )(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


# 初始化数据
inouts = tf.keras.Input([229, 229, 3])
x = conv2d(inouts, 32, 3, 3, strides=(2, 2), padding='valid')
x = conv2d(x, 32, 3, 3, padding='valid')
x = conv2d(x, 64, 3, 3)
x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2))(x)

x = conv2d(x, 80, 3, 3, padding='valid')
x = conv2d(x, 192, 3, 3, padding='valid')
x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2))(x)

# 第一部分

# 第一条支路
branch_1 = conv2d(x, 64, 1, 1)
# 第二条支路
branch_2 = conv2d(x, 48, 1, 1)
branch_2 = conv2d(branch_2, 64, 5, 5)
# 第三条支路
branch_3 = conv2d(x, 64, 1, 1)
branch_3 = conv2d(branch_3, 96, 3, 3)
branch_3 = conv2d(branch_3, 96, 3, 3)
# 第四条支路
branch_4 = tf.keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_4 = conv2d(branch_4, 32, 1, 1)
# 支路合并
x = tf.keras.layers.concatenate([branch_1, branch_2, branch_3, branch_4], axis=3)

# 第二部分

# 第一条支路
branch_1 = conv2d(x, 64, 1, 1)
# 第二条支路
branch_2 = conv2d(x, 48, 1, 1)
branch_2 = conv2d(branch_2, 64, 5, 5)
# 第三条支路
branch_3 = conv2d(x, 64, 1, 1)
branch_3 = conv2d(branch_3, 96, 3, 3)
branch_3 = conv2d(branch_3, 96, 3, 3)
# 第四条支路
branch_4 = tf.keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_4 = conv2d(branch_4, 64, 1, 1)
# 支路合并
x = tf.keras.layers.concatenate([branch_1, branch_2, branch_3, branch_4], axis=3)

# 第三部分

# 第一条支路
branch_1 = conv2d(x, 64, 1, 1)
# 第二条支路
branch_2 = conv2d(x, 48, 1, 1)
branch_2 = conv2d(branch_2, 64, 5, 5)
# 第三条支路
branch_3 = conv2d(x, 64, 1, 1)
branch_3 = conv2d(branch_3, 96, 3, 3)
branch_3 = conv2d(branch_3, 96, 3, 3)
# 第四条支路
branch_4 = tf.keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_4 = conv2d(branch_4, 64, 1, 1)
# 支路合并
x = tf.keras.layers.concatenate([branch_1, branch_2, branch_3, branch_4], axis=3)

# 第四部分

# 第一条支路
branch_1 = conv2d(x, 384, 3, 3, strides=(2, 2), padding='valid')
# 第二条之路
branch_2 = conv2d(x, 64, 1, 1)
branch_2 = conv2d(branch_2, 96, 3, 3)
branch_2 = conv2d(branch_2, 96, 3, 3, strides=(2, 2), padding='valid')
# 第三条支路
branch_3 = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2))(x)
# 支路合并
x = tf.keras.layers.concatenate([branch_1, branch_2, branch_3], axis=3)

# 第五部分

# 第一条支路
branch_1 = conv2d(x, 192, 1, 1)
# 第二条支路
branch_2 = conv2d(x, 128, 1, 1)
branch_2 = conv2d(branch_2, 128, 1, 7)
branch_2 = conv2d(branch_2, 128, 7, 1)
# 第三条支路
branch_3 = conv2d(x, 128, 1, 1)
branch_3 = conv2d(branch_3, 128, 7, 1)
branch_3 = conv2d(branch_3, 128, 1, 7)
branch_3 = conv2d(branch_3, 128, 7, 1)
branch_3 = conv2d(branch_3, 128, 1, 7)
# 第四条支路
branch_4 = tf.keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_4 = conv2d(branch_4, 192, 1, 1)
# 支路合并
x = tf.keras.layers.concatenate([branch_1, branch_2, branch_3, branch_4], axis=3)

# 第六,七部分相同

for i in range(2):
    # 第一条支路
    branch_1 = conv2d(x, 192, 1, 1)
    # 第二条支路
    branch_2 = conv2d(x, 160, 1, 1)
    branch_2 = conv2d(branch_2, 160, 1, 7)
    branch_2 = conv2d(branch_2, 192, 7, 1)
    # 第三条支路
    branch_3 = conv2d(x, 160, 1, 1)
    branch_3 = conv2d(branch_3, 160, 7, 1)
    branch_3 = conv2d(branch_3, 160, 1, 7)
    branch_3 = conv2d(branch_3, 160, 7, 1)
    branch_3 = conv2d(branch_3, 192, 1, 7)
    # 第四条支路
    branch_4 = tf.keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_4 = conv2d(branch_4, 192, 1, 1)
    # 支路合并
    x = tf.keras.layers.concatenate([branch_1, branch_2, branch_3, branch_4], axis=3)

# 第八部分

# 第一条支路
branch_1 = conv2d(x, 192, 1, 1)
# 第二条支路
branch_2 = conv2d(x, 192, 1, 1)
branch_2 = conv2d(branch_2, 192, 1, 7)
branch_2 = conv2d(branch_2, 192, 7, 1)
# 第三条支路
branch_3 = conv2d(x, 192, 1, 1)
branch_3 = conv2d(branch_3, 192, 7, 1)
branch_3 = conv2d(branch_3, 192, 1, 7)
branch_3 = conv2d(branch_3, 192, 7, 1)
branch_3 = conv2d(branch_3, 192, 1, 7)
# 第四条支路
branch_4 = tf.keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_4 = conv2d(branch_4, 192, 1, 1)
# 支路合并
x = tf.keras.layers.concatenate([branch_1, branch_2, branch_3, branch_4], axis=3)

# 第九部分

# 第一条支路
branch_1 = conv2d(x, 192, 1, 1)
branch_1 = conv2d(branch_1, 320, 3, 3, strides=(2, 2), padding='valid')
# 第二条支路
branch_2 = conv2d(x, 192, 1, 1)
branch_2 = conv2d(branch_2, 192, 1, 7)
branch_2 = conv2d(branch_2, 192, 7, 1)
branch_2 = conv2d(branch_2, 192, 3, 3, strides=(2, 2), padding='valid')
# 第三条支路
branch_pool = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2))(x)
# 支路合并
x = tf.keras.layers.concatenate([branch_1, branch_2, branch_pool], axis=3)

# 第十部分

# 第一条支路
branch3x3 = conv2d(x, 192, 1, 1)
branch3x3 = conv2d(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')
# 第二条支路
branch7x7x3 = conv2d(x, 192, 1, 1)
branch7x7x3 = conv2d(branch7x7x3, 192, 1, 7)
branch7x7x3 = conv2d(branch7x7x3, 192, 7, 1)
branch7x7x3 = conv2d(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')
# 第三条支路
branch_pool = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2))(x)
# 支路合并
x = tf.keras.layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=3)

# 第十一,十二部分相同

for i in range(2):
    branch1x1 = conv2d(x, 320, 1, 1)

    branch3x3 = conv2d(x, 384, 1, 1)
    branch3x3_1 = conv2d(branch3x3, 384, 1, 3)
    branch3x3_2 = conv2d(branch3x3, 384, 3, 1)
    branch3x3 = tf.keras.layers.concatenate([branch3x3_1, branch3x3_2], axis=3)

    branch3x3dbl = conv2d(x, 448, 1, 1)
    branch3x3dbl = conv2d(branch3x3dbl, 384, 3, 3)
    branch3x3dbl_1 = conv2d(branch3x3dbl, 384, 1, 3)
    branch3x3dbl_2 = conv2d(branch3x3dbl, 384, 3, 1)
    branch3x3dbl = tf.keras.layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

    branch_pool = tf.keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d(branch_pool, 192, 1, 1)
    x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3)

x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
x = tf.keras.layers.Dense(1000, activation='softmax', name='predictions')(x)

model = tf.keras.Model(inouts, x)
model.summary()
