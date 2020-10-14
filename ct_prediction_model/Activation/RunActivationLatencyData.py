# coding: utf-8
import time
import psutil
from keras.layers import Input
from keras.layers import Activation
from keras.models import Model
from keras import backend as K
import numpy as np
import pandas as pd

def get_only_act_model(activation_type,input_dim):
    print('create model ...')
    inputs = Input(shape=(input_dim,))

    x = Activation(activation_type)(inputs)

    # Create model.
    model = Model(inputs, x)
    return model

def run_once_act(data_dict,activation_type,input_dim):

    model = get_only_act_model(activation_type=activation_type,input_dim=input_dim)
    

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

    mem_total = data.total / 1024 / 1024  # 总内存,单位为byte/ 1024 / 1024 = Mb

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
    data_dict['activation_type'].append(activation_type)


def run_once_act_inputshape(data_dict, activation_type, input_shape):

    def get_inputshape_act_model(activation_type, input_shape):
        print('create model ...')
        inputs = Input(shape=input_shape)

        x = Activation(activation_type)(inputs)

        # Create model.
        model = Model(inputs, x)
        return model

    model = get_inputshape_act_model(activation_type=activation_type, input_shape=input_shape)


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

    data_dict['activation_type'].append(activation_type)


    data_dict['out_dim'].append(input_shape[0])
    data_dict['last_deep_dim'].append(input_shape[1])

def run_once():
    input_dim = 10
    activation_type = 'relu'
    print('create model ...')
    inputs = Input(shape=(input_dim,))

    x = Activation(activation_type)(inputs)

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


        'activation_type': [],
        'out_dim': [],
        'last_deep_dim': [],
    }


    for last_deep_dim in [16,32,64,96,256,480,512] + \
                         [10, 100, 128, 200, 256, 384, 500, 512, 640, 768, 896, 1000,1024]:
        for out_dim in [16,32,64,96,256,480,512] + \
                       [10,100,128,200,256,384,500,512,640,768,896,1000,1024]:
            input_shape = (out_dim ,last_deep_dim)
            print(input_shape)
            for activation_type in ['softmax', 'elu', 'selu', 'softplus',
                                    'softsign','relu', 'tanh', 'sigmoid',
                                    'hard_sigmoid']:
                print(activation_type)
                for i in range(1):
                    run_once_act_inputshape(data_dict,activation_type,input_shape)

    data = pd.DataFrame(data_dict)
    print('shape ', data.shape)
    print(data.head())
    data.to_csv('activation_train.csv', index=False, encoding='utf-8')
    print('write end ...')