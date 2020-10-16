# coding=utf-8
"""
Tensorflow graph 冷冻机
转换Tensorflow训练模型为.pb

Code adapted from:
https://gist.github.com/morgangiraud/249505f540a5e53a48b0c1a869d370bf#file-medium-tffreeze-1-py
"""
import os
from functools import reduce
import tensorflow as tf
from tensorflow.python.framework import graph_util
from keras.models import load_model
import json
from Utils import get_model_dependence

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def gen_freeze_graph_node_name(model_folder, output_path="frozen_model_node.txt",
                               output_node_names="dense_5/Softmax,dense_2/Softmax,dense_4/Softmax"):
    # 我们获取检查点全路径
    try:
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path
        print("[INFO] input_checkpoint:", input_checkpoint)
    except:
        input_checkpoint = model_folder
        print("[INFO] Model folder", model_folder)

    '''
    # 在导出graph之前，我们需要精确确定output node
    # 这就是TF决定他必须保留Graph的哪一部分以及可以丢弃哪一部分的方式
    output_node_names = "dense_5/Softmax"  # NOTE: 改变这里!!!
    '''

    # 我们清除设备以允许TensorFlow控制它将在哪个设备上加载操作
    clear_devices = True

    # 我们导入meta graph并取回一个Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # 我们取回protobuf图定义
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # 我们开始一个session并且保存图权重
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # 我们使用内置的TF helper将变量导出为常量
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,  # 该session用于检索权重
            input_graph_def,  # sgraph_def用于检索节点
            output_node_names.split(",")  # 输出节点名称用于选择有用的节点
        )

        # 将推理的操作节点保存到文件
        with open(output_path,'w',encoding='utf-8') as wf:
            for op in output_graph_def.node:
                print(op.name)
                wf.write(op.name + '\n')

        print("[INFO] output_graph:", output_path)
        print("[INFO] all done")

def gen_android_node_json(infer_node_path,out_info_path,input_shape=[32 * 32 * 3]):
    # 加载模型
    TargetNet = load_model('model.h5')

    # 取得层名
    layer_names = [layer.name for layer in TargetNet.layers]
    print('layer_names ',len(layer_names),layer_names[:5])
    # 层shape
    layer_shapes = [[1]+list(layer.output_shape[1:]) for layer in TargetNet.layers]
    print('layer_shapes ',len(layer_shapes),layer_shapes[:5])
    # 层大小
    layer_out_sizes = [reduce(lambda x, y: x * y, layer.output_shape[1:], 1) for layer in TargetNet.layers]
    print('layer_out_sizes ', len(layer_out_sizes),layer_out_sizes[:5])

    # 加载推理操作名
    ops = open(infer_node_path,'r',encoding='utf-8').readlines()
    ops = [x.strip() for x in ops]
    print('ops ',ops[:5])

    # 取得每层的最后操作名
    last_ops = []
    for name in layer_names:
        # print('name',name)
        match_ops = []
        for op in ops:
            # first_name = op.split('/')  # [0]
            # if name in first_name:
            #     # print('first_name', first_name)
            #     match_ops.append(op)
            first_name = op.split('/')[0]
            if name == first_name:
                # print('first_name', first_name)
                match_ops.append(op)
        last_ops.append(match_ops[-1])

    print('len last_ops ',len(last_ops),last_ops[:5])

        # 加载依赖关系
    layer_dependences = get_model_dependence()
    print('layer_dependences ',len(layer_dependences),layer_dependences[:5])

    # 加载层名、层输出大小、层shape、层面到index的映射
    results = []
    name2index = {}
    for i in range(len(TargetNet.layers)):

        layer =  TargetNet.layers[i]

        name2index[layer.name] = i

        bean = dict()
        if i == 0:
            bean['layerName'] = layer.name
            bean['opName'] = last_ops[i]
            bean["previousOpName"] = [layer.name]
            bean["previousSize"] = input_shape
            bean["previousShape"] = [[1, *input_shape]]
            bean['outSize'] = int(reduce(lambda x, y: x * y, input_shape, 1))
            bean['outShape'] = [1, *input_shape]
        else:
            bean['layerName'] = layer.name
            bean['opName'] = last_ops[i]
            bean["previousOpName"] = [last_ops[name2index[x[0]]] for x in layer_dependences[i]]
            bean["previousSize"] = [layer_out_sizes[name2index[x[0]]] for x in layer_dependences[i]]
            bean["previousSize"] = [int(x) for x in bean["previousSize"]]
            bean["previousShape"] = [layer_shapes[name2index[x[0]]] for x in layer_dependences[i]]
            bean["previousShape"] = [[int(a) for a in x] for x in bean["previousShape"]]
            bean['outSize'] = int(layer_out_sizes[i])
            bean['outShape'] = [int(x) for x in layer_shapes[i]]

        print(bean)
        results.append(bean)

    with open(out_info_path, 'w') as f:
        json.dump(results, f)

def gen_out_size(out_info_path):
    # 加载模型
    TargetNet = load_model('model.h5')

    # 层大小
    layer_out_sizes = [reduce(lambda x, y: x * y, layer.output_shape[1:], 1) for layer in TargetNet.layers]
    print('layer_out_sizes ', len(layer_out_sizes), layer_out_sizes[:5])

    results = {'layerOutSizes':list(set([int(x) for x in layer_out_sizes]))}

    with open(out_info_path, 'w') as f:
        json.dump(results, f)

def gen_down_up_time(outsize_path,time_path):
    # outsize
    f = open(outsize_path, encoding='utf-8')
    layerOutSizes = json.load(f)['layerOutSizes']
    print(layerOutSizes)
    with open( time_path,'w') as wf:
        for size in layerOutSizes:
            wf.write(str(size) + '\t' + str(0) +'\n')

if __name__ == '__main__':
    ''' meta 转 pb '''
    gen_freeze_graph_node_name('out/', 'frozen_googlenet_node.txt')

    gen_android_node_json('frozen_googlenet_node.txt','googlenet_node.json')

    gen_out_size('googlenet_outsize.json')
