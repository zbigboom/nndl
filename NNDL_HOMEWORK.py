# 环境pycharm，anaconda3，python3.7，tensorflow2.1
from tensorflow import keras

# AlexNet
'''
# 定义输入的数据格式大小为224*224*3
inputs = keras.Input(shape=(224, 224, 3), )
# 第一个卷积层11*11*3*96，步长为4，padding=4得到55*55*96的特征映射组
conv_1 = keras.layers.Conv2D(filters=96,
                             kernel_size=11,
                             strides=4,
                             activation='relu',
                             padding='same')(inputs)
# 第一个汇聚层3*3步长为2，得到27*27*96的特征映射组
pool_1 = keras.layers.MaxPool2D(pool_size=3,
                                strides=2)(conv_1)
# 第二个卷积层5*5*48*256，步长为1，padding=2得到27*27*128的特征映射组
conv_2 = keras.layers.Conv2D(filters=256,
                             kernel_size=5,
                             strides=1,
                             activation='relu',
                             padding='same')(pool_1)
# 第二个汇聚层3*3步长为2，得到13*13*256的特征映射组
pool_2 = keras.layers.MaxPool2D(pool_size=3,
                                strides=2)(conv_2)
# 第三个卷积层3*3*256*384，步长为1，padding=1得到13*13*384的特征映射组
conv_3 = keras.layers.Conv2D(filters=384,
                             kernel_size=3,
                             strides=1,
                             activation='relu',
                             padding='same')(pool_2)
# 第四个卷积层3*3*192*384，步长为1，padding=1得到13*13*384的特征映射组
conv_4 = keras.layers.Conv2D(filters=384,
                             kernel_size=3,
                             strides=1,
                             activation='relu',
                             padding='same')(conv_3)
# 第五个卷积层3*3*192*256，步长为1，padding=1得到13*13*256的特征映射组
conv_5 = keras.layers.Conv2D(filters=256,
                             kernel_size=5,
                             strides=1,
                             activation='relu',
                             padding='same')(conv_4)
# 第三个汇聚层3*3，步长为2得到6*6*256的特征映射组
pool_3 = keras.layers.MaxPool2D(pool_size=3,
                                strides=2)(conv_5)
# 拉直，将数据拉直
flatten = keras.layers.Flatten()(pool_3)
# 第一个全链接层4096个神经元，激活函数为relu
FC1 = keras.layers.Dense(4096, activation='relu')(flatten)
# 第二个全链接层4096个神经元，激活函数为relu
FC2 = keras.layers.Dense(4096, activation='relu')(FC1)
# 第三个全链接层1000个神经元，激活函数为softmax
FC3 = keras.layers.Dense(1000, activation='softmax')(FC2)
# 构造模型
model = keras.Model(inputs, FC3)
# 模型可视化
model.summary()
'''
# GoogLeNet-V3
'''
# 定义卷积操作
def conv2d(x,
           filters,
           num_row,
           num_col,
           padding='same',
           strides=(1, 1),
           ):
    
    # 卷积层
    x = keras.layers.Conv2D(filters=filters,
                            kernel_size=(num_row, num_col),
                            strides=strides,
                            padding=padding,
                            use_bias=False
                            )(x)
    # 规范化输入缩放和激活操作
    x = keras.layers.BatchNormalization(scale=False)(x)
    # 激活函数
    x = keras.layers.Activation('relu')(x)
    return x


# 初始化数据
inouts = keras.Input(shape=(229, 229, 3))
x = conv2d(inouts, 32, 3, 3, strides=(2, 2), padding='valid')
x = conv2d(x, 32, 3, 3, padding='valid')
x = conv2d(x, 64, 3, 3)
x = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(x)

x = conv2d(x, 80, 3, 3, padding='valid')
x = conv2d(x, 192, 3, 3, padding='valid')
x = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(x)

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
branch_4 = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_4 = conv2d(branch_4, 32, 1, 1)
# 支路合并
x = keras.layers.concatenate([branch_1, branch_2, branch_3, branch_4], axis=3)

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
branch_4 = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_4 = conv2d(branch_4, 64, 1, 1)
# 支路合并
x = keras.layers.concatenate([branch_1, branch_2, branch_3, branch_4], axis=3)

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
branch_4 = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_4 = conv2d(branch_4, 64, 1, 1)
# 支路合并
x = keras.layers.concatenate([branch_1, branch_2, branch_3, branch_4], axis=3)

# 第四部分

# 第一条支路
branch_1 = conv2d(x, 384, 3, 3, strides=(2, 2), padding='valid')
# 第二条之路
branch_2 = conv2d(x, 64, 1, 1)
branch_2 = conv2d(branch_2, 96, 3, 3)
branch_2 = conv2d(branch_2, 96, 3, 3, strides=(2, 2), padding='valid')
# 第三条支路
branch_3 = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(x)
# 支路合并
x = keras.layers.concatenate([branch_1, branch_2, branch_3], axis=3)

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
branch_4 = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_4 = conv2d(branch_4, 192, 1, 1)
# 支路合并
x = keras.layers.concatenate([branch_1, branch_2, branch_3, branch_4], axis=3)

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
    branch_4 = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_4 = conv2d(branch_4, 192, 1, 1)
    # 支路合并
    x = keras.layers.concatenate([branch_1, branch_2, branch_3, branch_4], axis=3)

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
branch_4 = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_4 = conv2d(branch_4, 192, 1, 1)
# 支路合并
x = keras.layers.concatenate([branch_1, branch_2, branch_3, branch_4], axis=3)

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
branch_pool = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(x)
# 支路合并
x = keras.layers.concatenate([branch_1, branch_2, branch_pool], axis=3)

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
branch_pool = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(x)
# 支路合并
x = keras.layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=3)

# 第十一,十二部分相同

for i in range(2):
    branch1x1 = conv2d(x, 320, 1, 1)

    branch3x3 = conv2d(x, 384, 1, 1)
    branch3x3_1 = conv2d(branch3x3, 384, 1, 3)
    branch3x3_2 = conv2d(branch3x3, 384, 3, 1)
    branch3x3 = keras.layers.concatenate([branch3x3_1, branch3x3_2], axis=3)

    branch3x3dbl = conv2d(x, 448, 1, 1)
    branch3x3dbl = conv2d(branch3x3dbl, 384, 3, 3)
    branch3x3dbl_1 = conv2d(branch3x3dbl, 384, 1, 3)
    branch3x3dbl_2 = conv2d(branch3x3dbl, 384, 3, 1)
    branch3x3dbl = keras.layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

    branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d(branch_pool, 192, 1, 1)
    x = keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3)

x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
x = keras.layers.Dense(1000, activation='softmax', name='predictions')(x)

# 构造模型
model = keras.Model(inouts, x)
# 模型可视化
model.summary()
'''
# GoogLeNetv1

# 定义卷积层
def conv2d(inputs,filters,kernel_size,padding='same',strides=1):
    x=keras.layers.Conv2D(filters=filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding)(inputs)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.Activation('relu')(x)
    return x
# 定义模块
def inception(inputs,filters):
    branch1x1=conv2d(inputs,filters,1)

    branch3x3=conv2d(inputs,filters,1)
    branch3x3=conv2d(branch3x3,filters,3)

    branch5x5=conv2d(inputs,filters,1)
    branch5x5=conv2d(branch5x5,filters,5)

    branchpool=keras.layers.MaxPool2D(pool_size=3,strides=1,padding='same')(inputs)
    branchpool=conv2d(branchpool,filters,1)


    x=keras.layers.concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=3)

    return x

# 输入数据
inputs=keras.Input(shape=(224,224,3),)
# 第一个卷积层7*7*64步长为2
x=conv2d(inputs,64,7,strides=2)
# 第一个汇聚层3*3步长为2
x=keras.layers.MaxPool2D(pool_size=3,strides=2)(x)
# 第二个卷积层1*1*64步长为1
x=conv2d(x,64,1)
# 第三个卷积层3*3*192步长为3
x=conv2d(x,192,3)
# 第二个汇聚层3*3步长为2
x=keras.layers.MaxPool2D(pool_size=3,strides=2)(x)
# 第一个模块
x=inception(x,64)
# 第二个模块
x=inception(x,128)
# 第三个汇聚层
x=keras.layers.MaxPool2D(pool_size=3,strides=2)(x)
# 第三个模块
x=inception(x,192)

# 这里输出softmax0

# 第四个模块
x=inception(x,160)
# 第五个模块
x=inception(x,128)
# 第六个模块
x=inception(x,112)

# 这里输出softmax1

# 第七个模块
x=inception(x,256)
# 第四个汇聚层
x=keras.layers.MaxPool2D(pool_size=3,strides=2)(x)
# 第八个模块
x=inception(x,256)
# 第九个模块
x=inception(x,256)
# 第五个汇聚层
x=keras.layers.AveragePooling2D(pool_size=7,strides=1,padding='same')(x)
# 全连接层
x=keras.layers.Dense(1000,activation='softmax')(x)

model=keras.models.Model(inputs,x)
model.summary()





# VGGNet-16
'''
# 定义输入数据的格式
inputs = keras.Input(shape=(224, 224, 3), )
# 第一个卷积层3*3*64，步长为1padding=2得到224*224*64的特征映射组
conv_1 = keras.layers.Conv2D(filters=64,
                             kernel_size=3,
                             strides=1,
                             activation='relu',
                             padding='same')(inputs)
# 第二个卷积层3*3*64，步长为1padding=2得到224*224*64的特征映射组
conv_2 = keras.layers.Conv2D(filters=64,
                             kernel_size=3,
                             strides=1,
                             activation='relu',
                             padding='same')(conv_1)
# 第一个汇聚层2*2步长为2得到112*112*128的特征映射组
pool_1 = keras.layers.MaxPool2D(pool_size=2,
                                strides=2)(conv_2)
# 第三个卷积层3*3*128，步长为3padding=2得到112*112*128的特征映射组
conv_3 = keras.layers.Conv2D(filters=128,
                             kernel_size=3,
                             strides=1,
                             activation='relu',
                             padding='same')(pool_1)
# 第四个卷积层3*3*128，步长为1padding=2得到112*112*128的特征映射组
conv_4 = keras.layers.Conv2D(filters=128,
                             kernel_size=3,
                             strides=1,
                             activation='relu',
                             padding='same')(conv_3)
# 第二个汇聚层2*2步长为2得到56*56*256的特征映射组
pool_2 = keras.layers.MaxPool2D(pool_size=2,
                                strides=2)(conv_4)
# 第五个卷积层3*3*256，步长为1padding=2得到56*56*256的特征映射组
conv_5 = keras.layers.Conv2D(filters=256,
                             kernel_size=3,
                             strides=1,
                             activation='relu',
                             padding='same')(pool_2)
# 第六个卷积层3*3*256，步长为1padding=2得到56*56*256的特征映射组
conv_6 = keras.layers.Conv2D(filters=256,
                             kernel_size=3,
                             strides=1,
                             activation='relu',
                             padding='same')(conv_5)
# 第七个卷积层3*3*256，步长为1padding=2得到56*56*256的特征映射组
conv_7 = keras.layers.Conv2D(filters=256,
                             kernel_size=3,
                             strides=1,
                             activation='relu',
                             padding='same')(conv_6)
# 第三个汇聚层2*2步长为2得到28*28*512的特征映射组
pool_3 = keras.layers.MaxPool2D(pool_size=2,
                                strides=2)(conv_7)
# 第八个卷积层3*3*512，步长为1padding=2得到28*28*512的特征映射组
conv_8 = keras.layers.Conv2D(filters=512,
                             kernel_size=3,
                             strides=1,
                             activation='relu',
                             padding='same')(pool_3)
# 第九个卷积层3*3*512，步长为1padding=2得到28*28*512的特征映射组
conv_9 = keras.layers.Conv2D(filters=512,
                             kernel_size=3,
                             strides=1,
                             activation='relu',
                             padding='same')(conv_8)
# 第十个卷积层3*3*512，步长为1padding=2得到28*28*512的特征映射组
conv_10 = keras.layers.Conv2D(filters=512,
                              kernel_size=3,
                              strides=1,
                              activation='relu',
                              padding='same')(conv_9)
# 第四个汇聚层2*2步长为2得到14*14*128的特征映射组
pool_4 = keras.layers.MaxPool2D(pool_size=2,
                                strides=2)(conv_10)
# 第十一个卷积层3*3*512，步长为1padding=2得到14*14*512的特征映射组
conv_11 = keras.layers.Conv2D(filters=512,
                              kernel_size=3,
                              strides=1,
                              activation='relu',
                              padding='same')(pool_4)
# 第十二个卷积层3*3*512，步长为1padding=2得到14*14*512的特征映射组
conv_12 = keras.layers.Conv2D(filters=512,
                              kernel_size=3,
                              strides=1,
                              activation='relu',
                              padding='same')(conv_11)
# 第十三个卷积层3*3*512，步长为1padding=2得到14*14*512的特征映射组
conv_13 = keras.layers.Conv2D(filters=512,
                              kernel_size=3,
                              strides=1,
                              activation='relu',
                              padding='same')(conv_12)
# 第五个汇聚层2*2步长为2得到7*7*512
pool_5 = keras.layers.MaxPool2D(pool_size=2,
                                strides=2)(conv_13)
# 拉直，将数据拉直
flatten = keras.layers.Flatten()(pool_5)
# 第一个全链接层4096个神经元，激活函数为relu
FC1 = keras.layers.Dense(4096, activation='relu')(flatten)
# 第二个全链接层4096个神经元，激活函数为relu
FC2 = keras.layers.Dense(4096, activation='relu')(FC1)
# 第三个全链接层1000个神经元，激活函数为softmax
FC3 = keras.layers.Dense(1000, activation='softmax')(FC2)
# 构造模型
model = keras.models.Model(inputs, FC3)
# 模型可视化
model.summary()
'''
# ResNet50
'''
# 定义identity block模块，输入和输出维度相同，可以串联，用于加深网络
def identity_block(inputs, size, filter):
    filters1, filters2, filters3 = filter


    x = keras.layers.Conv2D(filters=filters1,
                            kernel_size=1)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters=filters2,
                            kernel_size=size,
                            padding='same')(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters=filters3,
                            kernel_size=1)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.add([x, inputs])
    x = keras.layers.Activation('relu')(x)
    return x

# 定义Conv Block模块，输入和输出的维度是不一样的，所以不能串联
def conv_block(inputs, size, filter , strides=2):
    filters1, filters2, filters3 = filter


    x = keras.layers.Conv2D(filters=filters1,
                            kernel_size=1,
                            strides=strides)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters=filters2,
                            kernel_size=size,
                            padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters=filters3,
                            kernel_size=1)(x)
    x = keras.layers.BatchNormalization()(x)
    # 短的路径
    shortcut = keras.layers.Conv2D(filters=filters3,
                                   kernel_size=1,
                                   strides=strides)(inputs)
    shortcut = keras.layers.BatchNormalization()(shortcut)
    # 路径合并
    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x


def ResNet50(inputs_shape=(224, 224, 3), classes=1000):
    inputs = keras.layers.Input(shape=inputs_shape)
    x = keras.layers.ZeroPadding2D((3, 3))(inputs)

    x = keras.layers.Conv2D(filters=64,
                            kernel_size=7,
                            strides=2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPool2D(pool_size=3,
                               strides=2)(x)

    x = conv_block(x, 3, [64, 64, 256], strides=1)
    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])

    x = conv_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])

    x = conv_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])

    x = conv_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])

    x = keras.layers.AveragePooling2D(pool_size=7,
                                      name='avg_pool')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(classes,
                           activation='softmax',
                           name='fc1000')(x)
    model = keras.models.Model(inputs, x)

    return model


model = ResNet50()
model.summary()
'''
