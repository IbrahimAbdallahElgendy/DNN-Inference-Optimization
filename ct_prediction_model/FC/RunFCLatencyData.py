# coding: utf-8
import time
import psutil
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
from keras import backend as K
import numpy as np
import pandas as pd

def get_only_fc_model(input_dim,
                      units,activation,use_bias):
    print('create model ...')
    inputs = Input(shape=(input_dim,))
    x = Dense(units=units, activation=activation,use_bias=use_bias)(inputs)
    # Create model.
    model = Model(inputs, x)
    return model

def run_once_fc(data_dict, input_dim,units,activation,use_bias):

    model = get_only_fc_model(input_dim,units,activation,use_bias)

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
    data_dict['output_dim'].append(output_shape[0])
    data_dict['neuro_units'].append(units)
    data_dict['activation_type'].append(activation)
    data_dict['use_bias'].append(use_bias)

def run_once():
    input_dim = 2048
    units = 512
    activation = 'relu'
    use_bias = False
    inputs = Input(shape=(input_dim,))
    x = Dense(units=units, activation=activation, use_bias=use_bias)(inputs)
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

        'input_dim': [],
        'output_dim': [],
        'neuro_units': [],
        'activation_type': [],
        'use_bias': []
    }

    for input_dim in [10, 100, 128, 200, 256, 300, 384, 512, 1024, 2048]:
        for units in [10, 100, 128, 200, 256, 300, 384, 512, 1000, 1024, 2048]:
            for activation in ['softmax', 'elu','selu','softplus','softsign',
                               'relu','tanh','sigmoid','hard_sigmoid',None]:
                for use_bias in [True, False]:
                    print(input_dim,units,activation,use_bias)
                    run_once_fc(data_dict,input_dim,units,activation,use_bias)


    data = pd.DataFrame(data_dict)
    print('shape ', data.shape)
    print(data.head())
    data.to_csv('fc_train_2_cpu10_cpu20.csv', index=False, encoding='utf-8')
    print('write end ...')
