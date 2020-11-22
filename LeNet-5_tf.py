import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist=tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
# 获取mnist数据集并作预处理

Train_images=tf.reshape(train_images,(train_images.shape[0],test_images.shape[1],train_images.shape[2],1))
print(Train_images.shape)
Test_images=tf.reshape(test_images,(test_images.shape[0],test_images.shape[1],test_images.shape[2],1))
print(Train_images)
# 输出训练集的特征大小
# reshape：给数据增加一个维度，使数据与网络结构匹配

# model = tf.keras.models.Sequential([网络结构])  #描述各层网络
# 拉直层：tf.keras.layers.Flatten() #拉直层可以变换张量的尺寸，把输入特征拉直为一维数组，是不含计算参数的层
#
# 全连接层：tf.keras.layers.Dense(神经元个数，
#
#                                                       activation = "激活函数“，
#
#                                                       kernel_regularizer = "正则化方式）
#
# 其中：activation可选 relu 、softmax、 sigmoid、 tanh等
#
#            kernel_regularizer可选 tf.keras.regularizers.l1() 、tf.keras.regularizers.l2()
#
# 卷积层：tf.keras.layers.Conv2D(filter = 卷积核个数，
#
#                                                    kernel_size = 卷积核尺寸，
#
#                                                    strides = 卷积步长,
#
#                                                    padding = ”valid“ or "same")
net=tf.keras.models.Sequential([
    # 第一层：6个5*5的卷积核，全填0；最大池化，2*2的池化核，步长为2，padding=‘VALID’
    tf.keras.layers.Conv2D(filters=6,kernel_size=5,activation='sigmoid',input_shape=(28,28,1),padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
    # 第二层
    tf.keras.layers.Conv2D(filters=16,kernel_size=5,activation='sigmoid',padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
    # 将28，28的数据拉直
     tf.keras.layers.Flatten(),
    # 三层全连接
    tf.keras.layers.Dense(120,activation='sigmoid'),
    tf.keras.layers.Dense(84,activation='sigmoid'),
    tf.keras.layers.Dense(10,activation='softmax')
])
# 损失函数和训练算法采用交叉熵函数(cross entropy)和批量随机梯度下降（SGD）
optimizer=tf.keras.optimizers.SGD(learning_rate=0.9,momentum=0.0,nesterov=False)
net.compile(optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
            )
# 训练模型5次，validation_split用来指定部分训练集为验证集
net.fit(Train_images,train_labels,epochs=5,validation_split=0.1)
# 预估准确率
test_loss,test_acc=net.evaluate(Test_images,test_labels)
print('\nTest accuracy:',test_acc)
# 对测试集图片进行预测
Predictions=net.predict(Test_images)
# 输出第一张图片的预测结果
print(Predictions[0])
print("The first picture's prediction is:",np.argmax(Predictions[0]))
print("the first picture is:",test_labels[0])
# 类别名称
class_names=['T-shirt/top','Trouser','Pullover','Dress','Cost','Sandal','Shirt','Sneaker','Bag','Ankle boot']
# 绘制前25个预测结果红色错误绿色正确
plt.figure(figsize=(15,15))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    predicted_label=np.argmax(Predictions[i])
    ture_label=test_labels[i]
    if predicted_label==ture_label:
        color='green'
    else:
        color='red'
    # 括号内为正确分类
    plt.xlabel("{}({})".format(class_names[predicted_label],class_names[ture_label]),color=color)
plt.show()
