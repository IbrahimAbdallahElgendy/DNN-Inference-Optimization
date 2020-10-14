# coding: utf-8
import time
import psutil
from keras.layers import Input
from keras.layers import MaxPooling2D,AveragePooling2D
from keras.models import Model
from keras import backend as K
import numpy as np
import pandas as pd

def get_only_pooling_model(input_shape,
                           pool_size,
                           strides,
                           padding, max_pool_flag):
    print('create model ...')
    inputs = Input(shape=input_shape)
    if max_pool_flag:
        x = MaxPooling2D(pool_size=pool_size,
                         strides=strides, padding=padding)(inputs)
    else:
        x = AveragePooling2D(pool_size=pool_size,
                         strides=strides, padding=padding)(inputs)

    # Create model.
    model = Model(inputs, x)
    return model

def run_once_pool(data_dict, input_shape, pool_size,
                  strides, padding, max_pool_flag):

    model = get_only_pooling_model(input_shape,pool_size=pool_size,
                           strides=strides,padding=padding, max_pool_flag=max_pool_flag)

    current_layer = model.layers[1]
    print('current_layer.name', current_layer.name)
    f_part = K.function([current_layer.input, K.learning_phase()],[current_layer.output])

    input_shape = model.layers[1].input_shape[1:]
    print('input_shape ', input_shape)
    input_data = np.random.rand(*input_shape)
    input_data = [np.asarray(input_data).reshape((1, *input_shape))]

    output_shape = model.layers[1].output_shape[1:]
    print('output_shape ', output_shape)

    layer_out = f_part(input_data + [0])[0]
    layer_out = f_part(input_data + [0])[0]

    data = psutil.virtual_memory()
    mem_total = data.total / 1024 / 1024
    mem_free = data.available / 1024 / 1024
    cpu_count = psutil.cpu_count()
    user_cpu_times, system_cpu_times, idle_cpu_times, interrupt_cpu_times, dpc_cpu_times = psutil.cpu_times()
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
    data_dict['system_cpu_times'].append(system_cpu_times)
    data_dict['idle_cpu_times'].append(idle_cpu_times)
    data_dict['interrupt_cpu_times'].append(interrupt_cpu_times)
    data_dict['dpc_cpu_times'].append(dpc_cpu_times)


    data_dict['input_width'].append(input_shape[0])
    data_dict['input_height'].append(input_shape[1])
    data_dict['input_channel'].append(input_shape[2])
    data_dict['output_width'].append(output_shape[0])
    data_dict['output_height'].append(output_shape[1])
    data_dict['output_channel'].append(output_shape[2])
    data_dict['pool_size_width'].append(pool_size[0])
    data_dict['pool_size_height'].append(pool_size[1])
    data_dict['strides_width'].append(strides[0])
    data_dict['strides_height'].append(strides[1])
    data_dict['padding_type'].append(padding)
    data_dict['max_pool_flag'].append(max_pool_flag)

def run_once():
    
    input_shape = (224, 224, 3)
    max_pool_flag = True
    pool_size_width, pool_size_height = (2,2)
    strides_width, strides_height = (2,2)
    padding = 'valid'
    print('create model ...')
    inputs = Input(shape=input_shape)
    if max_pool_flag:
        x = MaxPooling2D(pool_size=(pool_size_width, pool_size_height),
                         strides=(strides_width, strides_height), padding=padding)(inputs)
    else:
        x = AveragePooling2D(pool_size=(pool_size_width, pool_size_height),
                             strides=(strides_width, strides_height), padding=padding)(inputs)
    # Create model.
    model = Model(inputs, x)

    current_layer = model.layers[1]
    print('current_layer.name', current_layer.name)
    f_part = K.function([current_layer.input, K.learning_phase()],
                        [current_layer.output])

    input_shape = model.layers[1].input_shape[1:]
    print('input_shape ', input_shape)
    input_data = np.random.rand(*input_shape)
    input_data = [np.asarray(input_data).reshape((1, *input_shape))]

    layer_out = f_part(input_data + [0])[0]
    layer_out = f_part(input_data + [0])[0]

    begin = time.time()
    layer_out = f_part(input_data + [0])[0]
    end = time.time()

    used_time = (end - begin) * 1000
    print('used time ', used_time, ' ms')
    
if __name__ == '__main__':
    repeats = 1
    data_dict = {
        'label':[],
        'mem_total':[],
        'mem_free':[],
        'cpu_count':[],
        'cpu_percent':[],
        'user_cpu_times':[],
        'system_cpu_times':[],
        'idle_cpu_times':[],
        'interrupt_cpu_times': [],
        'dpc_cpu_times': [],
        'input_width':[],
        'input_height':[],
        'input_channel':[],
        'output_width': [],
        'output_height': [],
        'output_channel': [],
        'pool_size_width': [],
        'pool_size_height':[],
        'strides_width':[],
        'strides_height':[],
        'padding_type':[],
        'max_pool_flag':[]
    }
    for width in [8,12,14,16,26,28,32,54,112]:
        for depth in [32,64,96,192,256,480,512]:
            input_shape = (width,width,depth)
            for pool_size in [(2, 2), (3, 3), (4, 4)]:
                for strides in [(1, 1), (2, 2), (3, 3), (4, 4)]:
                    for padding in ['valid', 'same']:
                        for max_pool_flag in [True, False]:
                            print(input_shape, pool_size, strides, padding, max_pool_flag)
                            run_once_pool(data_dict, input_shape, pool_size,
                                          strides, padding, max_pool_flag)
    data = pd.DataFrame(data_dict)
    print('shape ', data.shape)
    print(data.head())
    data.to_csv('pool_train.csv', index=False, encoding='utf-8')
    # data.to_csv('pool_test.csv', index=False, encoding='utf-8')
    print('write end ...')
