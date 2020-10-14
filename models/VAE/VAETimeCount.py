# coding=utf-8
from keras.models import load_model
from keras import backend as K
import numpy as np
import time
from models.VAE.VAEModel import VAENet

def run_server_infer_time(input_shape=[32, 32, 3], times=200):
    # 加载 Keras model
    # TargetNet = load_model('model.h5')
    # 1.搭建模型结构
    TargetNet = VAENet().net

    print('Model is ready')
    # 输入
    test_x = np.random.rand(*input_shape)
    test_x = np.asarray(test_x).reshape((1, *input_shape))
    # 预测
    begin = time.time()
    pre_y = TargetNet.predict(test_x)
    pre_y = TargetNet.predict(test_x)
    end = time.time()
    print('used time ', (end - begin) * 1000, ' ms')
    # 统计
    sum_time = 0.0
    for i in range(times):
        begin = time.time()
        pre_y = TargetNet.predict(test_x)
        end = time.time()

        sum_time += (end - begin)
        print(i,' used time ', (end - begin) * 1000, ' ms')

    print('='*20)
    print('mean used time ', (sum_time)/times * 1000, ' ms')

def every_layer_runtime(times=200):

    # 每层的平均计算延迟
    mean_layer_runtime = []

    # 1.加载 Keras model
    TargetNet = load_model('model.h5')
    # 打印总层数
    layer_num = len(TargetNet.layers)
    print('layer_num ', layer_num)

    # 2.获取模型节点依赖关系
    layer_dependences = get_model_dependence()

    # 3.构造数据输入
    layer_results = {} # 保存每层的输出结果
    for num in range(layer_num):

        current_layer = TargetNet.layers[num]

        # 处理模型&输入
        if num == 0:
            ''' 第一层'''
            # 当前模型生成
            f_part = K.function([current_layer.input, K.learning_phase()],
                                [current_layer.output])
            print('Model is ready')
            # 输入数据构造
            input_shape = TargetNet.layers[0].input_shape[1:]
            print('input_shape ', input_shape)
            input_data = np.random.rand(*input_shape)
            input_data = [np.asarray(input_data).reshape((1, *input_shape))]
        else:
            '''非第一层'''
            pre_nodes = layer_dependences[num]

            if len(pre_nodes) == 1:
                '''只有一个前驱节点'''
                # 当前模型生成
                f_part = K.function([current_layer.input, K.learning_phase()],
                                    [current_layer.output])
                print('Model is ready')
                # 输入数据构造
                pre_node_name = pre_nodes[0][0]
                input_data = [layer_results[pre_node_name]]
            else:
                '''多个前驱节点'''
                # 当前模型生成
                f_part = K.function( current_layer.input + [K.learning_phase()],
                                    [current_layer.output])
                print('Model is ready')
                # 输入数据构造
                input_data = []
                for pre_node in pre_nodes:
                    pre_node_name = pre_node[0]
                    print(pre_node_name,layer_results[pre_node_name].shape)
                    input_data.append(layer_results[pre_node_name])

        # 4.预先执行两次
        layer_out = f_part(input_data + [0])[0]
        layer_out = f_part(input_data + [0])[0]

        # 5.开始统计计算时间
        sum_time = 0.0
        for i in range(times):
            begin = time.time()

            layer_out = f_part(input_data + [0])[0]

            end = time.time()

            sum_time += (end - begin)
            print(num ,i, ' used time ', (end - begin) * 1000, ' ms')

        # 保存层执行结果
        layer_results[current_layer.name] = layer_out

        mean_time = (sum_time) / times * 1000
        print(current_layer.name,'mean used time ', mean_time, ' ms')

        mean_layer_runtime.append(mean_time)
        print('='*20)
    return mean_layer_runtime


def get_layer_names():
    # 加载 Keras model
    TargetNet = load_model('model.h5')

    return [x.name for x in TargetNet.layers]

def get_model_dependence():

    layer_dependences = [] # [(前驱节点名,inbound_node_index,inbound_tensor_index)...]

    # 加载 Keras model
    TargetNet = load_model('model.h5')

    # 遍历layer
    for layer in TargetNet.layers:
        previous_nodes = []

        # 遍历前驱节点
        for node in layer._inbound_nodes:
            for i in range(len(node.inbound_layers)):
                inbound_layer = node.inbound_layers[i].name
                inbound_node_index = node.node_indices[i]
                inbound_tensor_index = node.tensor_indices[i]
                previous_nodes.append((inbound_layer,inbound_node_index,inbound_tensor_index))

        layer_dependences.append(previous_nodes)


    # 打印最终结果
    for i in range(len(layer_dependences)):
        print(i,layer_dependences[i])

    return layer_dependences

if __name__ == '__main__':
    run_server_infer_time(input_shape=[32*32])
    #
    # get_model_dependence()
    #
    # ''' 统计边缘服务器计算时间 '''
    # # 每层的名字
    # layer_names = get_layer_names()
    # # 每层的计算时间
    # mean_layer_runtime = every_layer_runtime()
    # # 保存到本地
    # assert len(layer_names) == len(mean_layer_runtime)
    # with open('EdgeNodeComputeTime.txt','w',encoding='utf-8') as wf:
    #     for i in range(len(layer_names)):
    #         wf.write(layer_names[i] + '\t' + ('%.4f' % mean_layer_runtime[i]) +'\n')
    # # 打印所有层计算时间的总和
    # print('sum time ',sum(mean_layer_runtime))








