# coding=utf-8
"""
视频识别的帧保留

Inspired by:
    Keras-Android-XOR => 如何在Android上运行keras
    https://github.com/OmarAflak/Keras-Android-XOR

    CNN-models
    https://github.com/km1414/CNN-models

构建和训练GoogLeNet模型，为移动设备输出冻结模型
"""
from models.GoogLeNet.helpers import *
import keras
from keras.layers import *
from keras.models import Model
from keras import optimizers
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import gc
from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

#
print('tf.__version__ ',tf.__version__)
print('keras.__version__',keras.__version__)

# 数据准备
# download_and_extract_data() # 下载数据
x_train, _, y_train= load_training_data() #
x_test, _, y_test = load_test_data()
input_shape = list(x_train.shape[1:])
print(input_shape)
classes = y_train.shape[1]
print(classes)


def export_model_for_mobile(model_name, input_node_name, output_node_name):
    ''' nodes.txt 里写入模型的节点名字 '''
    f1= open("nodes.txt","w+")
    for n in tf.get_default_graph().as_graph_def().node:
        f1.write(n.name)
        f1.write("\n")
    f1.close()

    ''' 将 模型图定义 写到文件 out/googlenet_graph.pbtxt
    将model保存在model.ph文件中
    然而使用的时候不仅要加载模型文件model.ph,还要加载保存的data.ckpt数据文件才能使用。这样保持了数据与模型的分离。
    '''
    tf.train.write_graph(K.get_session().graph_def, 'out', \
        model_name + '_graph.pbtxt')

    ''' 
    将 session 写到文件 out/googlenet.chkp.*** 
    .meta文件保存了当前图结构；.index文件保存了当前参数名；.data文件保存了当前参数值。
    '''
    tf.train.Saver().save(K.get_session(), 'out/' + model_name + '.chkp')


    """
    冷冻，可理解为整合合并；就是将模型文件和权重文件整合合并为一个文件，主要用途是便于发布
    把图和参数结构一起写到 frozen_googlenet.pb
    """
    freeze_graph.freeze_graph('out/' + model_name + '_graph.pbtxt', None, \
        False, 'out/' + model_name + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + model_name + '.pb', True, "")

    # 读取freeze的结果
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + model_name + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    """
    调用optimize_for_inference脚本会删除输入和输出节点之间所有不需要的节点。
    同时该脚本还做了一些其他优化以提高运行速度。
    例如它把显式批处理标准化运算跟卷积权重进行了合并，从而降低了计算量。
    """
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, [input_node_name], [output_node_name],
            tf.float32.as_datatype_enum)
    # 写到本地
    with tf.gfile.FastGFile('out/tensorflow_lite_' + model_name + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())


"""定义GoogleNet中的Inception module"""
# Inception module - main building block
def inception_module(X, filter_sizes):

    # 1x1 covolution
    conv_1x1 = Conv2D(filter_sizes[0], kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(X)

    # Bottleneck layer and 3x3 convolution
    conv_3x3 = Conv2D(filter_sizes[1], kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(X)
    conv_3x3 = Conv2D(filter_sizes[2], kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(conv_3x3)

    # Bottleneck layer and 5x5 convolution
    conv_5x5 = Conv2D(filter_sizes[3], kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(X)
    conv_5x5 = Conv2D(filter_sizes[4], kernel_size=5, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(conv_5x5)

    # Max pooling and bottleneck layer
    max_pool = MaxPooling2D(pool_size=3, strides=1, padding='same')(X)
    max_pool = Conv2D(filter_sizes[5], kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(max_pool)

    # Concatenate all tensors to 1 tensor
    X = concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool], axis=3)

    return X



# 辅助分类器-用于中间阶段的预测
def aux_classifier(X):

    # Average pooling, fc, dropout, fc
    X = AveragePooling2D(pool_size=3, strides=2, padding='same')(X)
    X = Conv2D(filters=128, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(0.0002))(X)
    X = Flatten()(X)
    X = Dense(1024, activation='relu', kernel_regularizer=l2(0.0002))(X)
    X = Dropout(0.7)(X)
    X = Dense(classes, activation='softmax', kernel_regularizer=l2(0.0002))(X)

    return X


"""GoogleNet模型"""
# 完整的模型
def create_model(input_shape, classes):

    # 定义输入
    X_input = Input(input_shape)

    # Stage 1 - layers before inception modules
    X = Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(X_input)
    X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Conv2D(filters=64, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(0.0002))(X)
    X = Conv2D(filters=192, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(X)
    X = BatchNormalization(axis=3)(X)
    X = MaxPooling2D(pool_size=3, strides=1, padding='same')(X)

    # Stage 2 - 2 inception modules and max pooling
    X = inception_module(X, filter_sizes=[64, 96, 128, 16, 32, 32])
    X = inception_module(X, filter_sizes=[128, 128, 192, 32, 96, 64])
    X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)

    # Stage 3 - 5 inception modules and max pooling
    X = inception_module(X, filter_sizes=[192, 96, 208, 16, 48, 64])
    aux_output_1 = aux_classifier(X)   # Auxiliary classifier
    X = inception_module(X, filter_sizes=[160, 112, 225, 24, 64, 64])
    X = inception_module(X, filter_sizes=[128, 128, 256, 24, 64, 64])
    X = inception_module(X, filter_sizes=[112, 144, 288, 32, 64, 64])
    aux_output_2 = aux_classifier(X) # Auxiliary classifier
    X = inception_module(X, filter_sizes=[256, 160, 320, 32, 128, 128])
    X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)

    # Stage 4 - 2 inception modules and average pooling
    X = inception_module(X, filter_sizes=[256, 160, 320, 32, 128, 128])
    X = inception_module(X, filter_sizes=[384, 192, 384, 48, 128, 128])
    X = AveragePooling2D(pool_size=4, strides=1, padding='valid')(X)

    # Stage 5 - dropout, linear fc, softmax fc
    X = Flatten()(X)
    X = Dropout(0.4)(X)
    X_output = Dense(classes, activation='softmax', kernel_regularizer=l2(0.0002))(X)

    # Create model - 结合主分类器和辅助分类器
    model = Model(inputs=X_input, outputs=[X_output, aux_output_1, aux_output_2])

    return model


# 定义optimizer并编译模型
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# 搭建模型结构
GoogLeNet = create_model(input_shape=input_shape, classes=classes)
# loss_weights - weighting losses of main classifier and auxiliary classifiers
GoogLeNet.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'], loss_weights=[1., 0.3, 0.3])


# 数据增强生成器
datagen = ImageDataGenerator(
            rotation_range=45,  # 随机旋转图像 in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # 水平随机移动图像(fraction of total width)
            height_shift_range=0.1,  # 垂直随机移动图像 (fraction of total height)
            horizontal_flip=True)  # 随机翻转图像



history = {'train_acc':[], 'train_loss':[], 'test_acc':[], 'test_loss':[]}
# 更改为较低的值以节省训练时间
epochs = 1


# 训练时间
for epoch in range(epochs):

    # 在完整数据集上处理数据增强
    x_train_aug, y_train_aug = datagen.flow(x_train, y_train,
                                            batch_size=len(x_train), shuffle=True).next()
    # 释放无用的内存
    _ = gc.collect()
    # 训练 1 epoch
    results = GoogLeNet.fit(x=x_train_aug, y=[y_train_aug, y_train_aug, y_train_aug],
                            epochs=1,
                            batch_size=250,
                            validation_data=(x_test, [y_test, y_test, y_test]),
                            verbose=0)

    # 保存 epoch 结果
    history['train_acc'].append(results.history['dense_5_acc'][0])
    history['train_loss'].append(results.history['dense_5_loss'][0])
    history['test_acc'].append(results.history['val_dense_5_acc'][0])
    history['test_loss'].append(results.history['val_dense_5_loss'][0])

    # 调度学习率变化
    lr = K.eval(GoogLeNet.optimizer.lr)
    if (epoch == 99) or (epoch == 199):
        K.set_value(GoogLeNet.optimizer.lr, lr/(1+1))

    # 打印 epoch 结果
    print('Epoch: '+str(epoch)+'/'+str(epochs-1), 'Learning rate:', lr,
          'Train_acc:', history['train_acc'][-1].round(4),
          'Train_loss:', history['train_loss'][-1].round(4),
          'Test_acc:', history['test_acc'][-1].round(4),
          'Test_loss:', history['test_loss'][-1].round(4))

# 将模型导出到移动设备
export_model_for_mobile('googlenet', "input_1", "dense_5/Softmax")

# 为server导出模型
GoogLeNet.save('model.h5')

# 绘图训练/验证结果
plot_results(history)

# 打印模型结构
GoogLeNet.summary()