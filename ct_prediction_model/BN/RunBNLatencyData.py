# coding: utf-8
import time
import psutil
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
import numpy as np
import pandas as pd
'''
创建bn层模型
'''
def get_only_bn_model(input_shape):
    print('create model ...')
    inputs = Input(shape=input_shape)
    x = BatchNormalization()(inputs)
    # Create model.
    model = Model(inputs, x)
    return model

def run_once_bn(data_dict, input_shape):

    model = get_only_bn_model(input_shape)

    # 当前模型生成
    current_layer = model.layers[1]
    print('current_layer.name', current_layer.name)
    f_part = K.function([current_layer.input, K.learning_phase()], [current_layer.output])

    # 创建输入变量
    input_shape = model.layers[1].input_shape[1:]
    print('input_shape ', input_shape)
    input_data = np.random.rand(*input_shape)
    input_data = [np.asarray(input_data).reshape((1, *input_shape))]

    # 输出大小
    output_shape = model.layers[1].output_shape[1:]
    print('output_shape ', output_shape)

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
    data_dict['output_width'].append(output_shape[0])
    data_dict['output_height'].append(output_shape[1])
    data_dict['output_channel'].append(output_shape[2])


def run_once():
    input_shape = (224, 224, 3)
    inputs = Input(shape=input_shape)
    x = BatchNormalization()(inputs)
    # Create model.
    model = Model(inputs, x)

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
    repeats = 1  # 不重复执行，负载会发生变化
    data_dict = {
        'label': [],
        'mem_total': [],
        'mem_free': [],
        'cpu_count': [],
        'cpu_percent': [],
        'user_cpu_times': [],
        'nice_cpu_times': [],
        'system_cpu_times': [],
        'idle_cpu_times': [],

        'input_width': [],
        'input_height': [],
        'input_channel': [],
        'output_width': [],
        'output_height': [],
        'output_channel': [],
    }
    for width in [5, 8, 12,14, 16, 26, 28, 32,48, 54,64,80,96,112,128,192,224,256,299]:
        for depth in [3,16,24,32,48,64,80,96,112,128,144,160,192,208,225,256,288,320,384,400,480,512]:
            input_shape = (width, width, depth)
            for i in range(3):
                print(input_shape)
                run_once_bn(data_dict, input_shape)

    # 将输出结果写到本地
    data = pd.DataFrame(data_dict)
    print('shape ', data.shape)
    print(data.head())
    data.to_csv('bn_train.csv', index=False, encoding='utf-8')
    print('write end ...')
