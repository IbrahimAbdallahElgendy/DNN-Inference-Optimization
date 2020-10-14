# coding=utf-8
from keras.models import load_model
from keras import backend as K
import numpy as np
import time

def run_server_infer_time(class_len=809, view_len=4, transf_param_len=12, times=200):

    TargetNet = load_model('model.h5')

    print('Model is ready')

    class_x = np.random.rand(class_len)
    class_x = np.asarray(class_x).reshape((1, class_len))
    view_x = np.random.rand(view_len)
    view_x = np.asarray(view_x).reshape((1, view_len))
    transf_x = np.random.rand(transf_param_len)
    transf_x = np.asarray(transf_x).reshape((1, transf_param_len))
    begin = time.time()
    pre_y = TargetNet.predict([class_x, transf_x,view_x])
    pre_y = TargetNet.predict([class_x, transf_x, view_x])
    end = time.time()
    print('used time ', (end - begin) * 1000, ' ms')
    sum_time = 0.0
    for i in range(times):
        begin = time.time()
        pre_y = TargetNet.predict([class_x, transf_x, view_x])
        end = time.time()

        sum_time += (end - begin)
        print(i,' used time ', (end - begin) * 1000, ' ms')

    print('='*20)
    print('mean used time ', (sum_time)/times * 1000, ' ms')

def every_layer_runtime(times=200):

    mean_layer_runtime = []

    TargetNet = load_model('model.h5')

    layer_num = len(TargetNet.layers)
    print('layer_num ', layer_num)

    layer_dependences = get_model_dependence()

    layer_results = {}
    for num in range(layer_num):

        current_layer = TargetNet.layers[num]

        if num < 3:

            f_part = K.function([current_layer.input, K.learning_phase()],
                                [current_layer.output])
            print('Model is ready')
            input_shape = TargetNet.layers[num].input_shape[1:]
            print('input_shape ', input_shape)
            input_data = np.random.rand(*input_shape)
            input_data = [np.asarray(input_data).reshape((1, *input_shape))]
        else:
            pre_nodes = layer_dependences[num]

            if len(pre_nodes) == 1:
                f_part = K.function([current_layer.input, K.learning_phase()],
                                    [current_layer.output])
                print('Model is ready')
                pre_node_name = pre_nodes[0][0]
                input_data = [layer_results[pre_node_name]]
            else:

                f_part = K.function( current_layer.input + [K.learning_phase()],
                                    [current_layer.output])
                print('Model is ready')
                input_data = []
                for pre_node in pre_nodes:
                    pre_node_name = pre_node[0]
                    print(pre_node_name,layer_results[pre_node_name].shape)
                    input_data.append(layer_results[pre_node_name])

        layer_out = f_part(input_data + [0])[0]
        layer_out = f_part(input_data + [0])[0]

        sum_time = 0.0
        for i in range(times):
            begin = time.time()

            layer_out = f_part(input_data + [0])[0]

            end = time.time()

            sum_time += (end - begin)
            print(num ,i, ' used time ', (end - begin) * 1000, ' ms')

        layer_results[current_layer.name] = layer_out

        mean_time = (sum_time) / times * 1000
        print(current_layer.name,'mean used time ', mean_time, ' ms')

        mean_layer_runtime.append(mean_time)
        print('='*20)
    return mean_layer_runtime


def get_layer_names():
    TargetNet = load_model('model.h5')

    return [x.name for x in TargetNet.layers]

def get_model_dependence():

    layer_dependences = []
    TargetNet = load_model('model.h5')

    for layer in TargetNet.layers:
        previous_nodes = []

        for node in layer._inbound_nodes:
            for i in range(len(node.inbound_layers)):
                inbound_layer = node.inbound_layers[i].name
                inbound_node_index = node.node_indices[i]
                inbound_tensor_index = node.tensor_indices[i]
                previous_nodes.append((inbound_layer,inbound_node_index,inbound_tensor_index))

        layer_dependences.append(previous_nodes)

    for i in range(len(layer_dependences)):
        print(i,layer_dependences[i])

    return layer_dependences

if __name__ == '__main__':

    get_model_dependence()

    layer_names = get_layer_names()

    mean_layer_runtime = every_layer_runtime()

    assert len(layer_names) == len(mean_layer_runtime)
    with open('EdgeNodeComputeTime.txt','w',encoding='utf-8') as wf:
        for i in range(len(layer_names)):
            wf.write(layer_names[i] + '\t' + ('%.4f' % mean_layer_runtime[i]) +'\n')
    print('sum time ',sum(mean_layer_runtime))
