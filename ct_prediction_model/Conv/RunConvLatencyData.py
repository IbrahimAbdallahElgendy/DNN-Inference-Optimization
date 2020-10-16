# coding: utf-8
import time
import psutil
from keras.layers import Input
from keras.layers import Conv2D
from keras.models import Model
from keras import backend as K
import numpy as np
import pandas as pd

'''
创建卷积层模型
'''
def get_only_conv_model(input_shape,filters,kernel_size,
                        strides,padding,
                        activation,use_bias):
    print('create model ...')
    img_input = Input(shape=input_shape)
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding,
                     activation=activation, input_shape=input_shape,use_bias=use_bias)(img_input)
    # Create model.
    model = Model(img_input, x)
    return model

def run_once_conv(data_dict, input_shape,filters,
                  kernel_size,strides,padding,
                  activation,use_bias):

    model = get_only_conv_model(input_shape, filters, kernel_size=kernel_size,
                                strides=strides, padding=padding,
                                activation=activation, use_bias=use_bias)
    
    # 当前模型生成
    current_layer = model.layers[1]
    print('current_layer.name', current_layer.name)
    f_part = K.function([current_layer.input, K.learning_phase()],[current_layer.output])

    # 创建输入变量
    input_shape = model.layers[1].input_shape[1:]
    print('input_shape ', input_shape)
    input_data = np.random.rand(*input_shape)
    input_data = [np.asarray(input_data).reshape((1, *input_shape))]

    # 预先执行两次，第一次运行会有准备工作
    layer_out = f_part(input_data + [0])[0]
    layer_out = f_part(input_data + [0])[0]

    # 系统信息
    data = psutil.virtual_memory()
    # 内存总数
    mem_total = data.total / 1024 / 1024  # 总内存,单位为byte/ 1024 / 1024 = Mb
    # 内存空闲数
    mem_free = data.available / 1024 / 1024
    # cpu数
    cpu_count = psutil.cpu_count()
    # cpu 时间
    user_cpu_times, nice_cpu_times, system_cpu_times, idle_cpu_times = psutil.cpu_times()
    # cpu利用率
    cpu_percent = psutil.cpu_percent(interval=1)

    used_time = 0.0
    for _ in range(repeats):
        start = time.time()
        layer_out = f_part(input_data + [0])[0]
        end = time.time()
        used_time += (end - start) * 1000

    used_time = used_time / repeats
    print('used time ', used_time)

    data_dict['label'].append(used_time)
    data_dict['mem_total'].append(mem_total)
    data_dict['mem_free'].append(mem_free)
    data_dict['cpu_count'].append(cpu_count)
    data_dict['cpu_percent'].append(cpu_percent)
    data_dict['user_cpu_times'].append(user_cpu_times)
    data_dict['nice_cpu_times'].append(nice_cpu_times)
    data_dict['system_cpu_times'].append(system_cpu_times)
    data_dict['idle_cpu_times'].append(idle_cpu_times)
    data_dict['input_width'].append(input_shape[0])
    data_dict['input_height'].append(input_shape[1])
    data_dict['input_channel'].append(input_shape[2])
    data_dict['filters'].append(filters)
    data_dict['kernel_size_width'].append(kernel_size[0])
    data_dict['kernel_size_height'].append(kernel_size[1])
    data_dict['strides_width'].append(strides[0])
    data_dict['strides_height'].append(strides[1])
    data_dict['padding_type'].append(padding)
    data_dict['activation_type'].append(activation)
    data_dict['use_bias'].append(use_bias)

def run_once():
    
    input_shape = (224, 224, 3)  # (32, 32, 3)
    img_input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)
    model = Model(img_input, x)

    # 当前模型生成
    current_layer = model.layers[1]
    print('current_layer.name', current_layer.name)
    f_part = K.function([current_layer.input, K.learning_phase()],
                        [current_layer.output])

    # 创建输入变量
    input_shape = model.layers[1].input_shape[1:]
    print('input_shape ', input_shape)
    input_data = np.random.rand(*input_shape)
    input_data = [np.asarray(input_data).reshape((1, *input_shape))]

    # 预先执行两次
    layer_out = f_part(input_data + [0])[0]
    layer_out = f_part(input_data + [0])[0]

    # 开始统计计算时间
    begin = time.time()
    layer_out = f_part(input_data + [0])[0]
    end = time.time()

    used_time = (end - begin) * 1000
    print('used time ', used_time, ' ms')
    
if __name__ == '__main__':
    repeats = 1 # 不重复执行，负载会发生变化
    data_dict = {
        'label':[],
        'mem_total':[],
        'mem_free':[],
        'cpu_count':[],
        'cpu_percent':[],
        'user_cpu_times':[],
        'nice_cpu_times':[],
        'system_cpu_times':[],
        'idle_cpu_times':[],
        'input_width':[],
        'input_height':[],
        'input_channel':[],
        'filters':[],
        'kernel_size_width': [],
        'kernel_size_height':[],
        'strides_width':[],
        'strides_height':[],
        'padding_type':[],
        'activation_type':[],
        'use_bias':[]
    }
    for input_shape in [(32,32,3),(64,64,3),(224, 224, 3), (299, 299, 3)]:
        for filters in [64,128,256,384,512]:
            for kernel_size in [(2, 2), (3, 3), (4, 4)]:
                for strides in [(1, 1), (2, 2), (3, 3)]:
                    for padding in ['valid', 'same']:
                        for activation in ['relu', 'sigmoid', 'tanh']:
                            for use_bias in [True, False]:
                                print(input_shape, filters,
                                      kernel_size, strides, padding,
                                      activation, use_bias)
                                run_once_conv(data_dict, input_shape, filters,
                                              kernel_size, strides, padding,
                                              activation, use_bias)

    # 将输出结果写到本地
    data = pd.DataFrame(data_dict)
    print('shape ', data.shape)
    print(data.head())
    data.to_csv('conv_train_cpu10_cpu20.csv', index=False, encoding='utf-8')
    print('write end ...')