# coding: utf-8
import time
import psutil
from keras.layers import Input
from keras.layers import Softmax
from keras.models import Model
from keras import backend as K
import numpy as np
import pandas as pd

def get_only_fc_model(input_dim):
    print('create model ...')
    inputs = Input(shape=(input_dim,))
    x = Softmax()(inputs)
    # Create model.
    model = Model(inputs, x)
    return model

def run_once_fc(data_dict, input_dim):

    model = get_only_fc_model(input_dim)

    current_layer = model.layers[1]
    print('current_layer.name', current_layer.name)
    f_part = K.function([current_layer.input, K.learning_phase()], [current_layer.output])

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

    data_dict['input_dim'].append(input_dim)


def run_once_fc_input_shape(data_dict,input_shape,
                            out_dim ,last_deep_dim):

    def get_inputshape_fc_model(input_shape):
        print('create model ...')
        inputs = Input(shape=input_shape)
        x = Softmax()(inputs)
        # Create model.
        model = Model(inputs, x)
        return model

    model = get_inputshape_fc_model(input_shape)

    current_layer = model.layers[1]
    print('current_layer.name', current_layer.name)
    f_part = K.function([current_layer.input, K.learning_phase()], [current_layer.output])

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
    data_dict['out_dim'].append(out_dim)
    data_dict['last_deep_dim'].append(last_deep_dim)

def run_once():
    input_dim = 2048
    inputs = Input(shape=(input_dim,))
    x = Softmax()(inputs)
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
        'label': [],
        'mem_total': [],
        'mem_free': [],
        'cpu_count': [],
        'cpu_percent': [],
        'user_cpu_times': [],
        'system_cpu_times': [],
        'idle_cpu_times': [],
        'interrupt_cpu_times': [],
        'dpc_cpu_times': [],
        'out_dim':[],
        'last_deep_dim':[],
    }
    """
    softmax_train_inputshape.csv
    """
    for last_deep_dim in [8,12,14,16,26,28,32,54,112] + [10,100,128,200,256,384,500,512,1000,1024]:
        for out_dim in [32,64,96,192,256,480,512] + [10,100,128,200,256,384,500,512,640,768,896,1000,
                                                     1024,1152, 1280,1408,2000,2048]:
            input_shape = (out_dim ,last_deep_dim)
            print(input_shape)
            for i in range(1):
                run_once_fc_input_shape(data_dict,input_shape, out_dim ,last_deep_dim)

    data = pd.DataFrame(data_dict)
    print('shape ', data.shape)
    print(data.head())
    # data.to_csv('softmax_train_1_more.csv', index=False, encoding='utf-8')
    data.to_csv('softmax_train_inputshape.csv', index=False, encoding='utf-8')
    print('write end ...')
