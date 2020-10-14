# coding=utf-8

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

    try:
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path
        print("[INFO] input_checkpoint:", input_checkpoint)
    except:
        input_checkpoint = model_folder
        print("[INFO] Model folder", model_folder)




    clear_devices = True


    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)


    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()


    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)


        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")
        )


        with open(output_path,'w',encoding='utf-8') as wf:
            for op in output_graph_def.node:
                print(op.name)
                wf.write(op.name + '\n')

        print("[INFO] output_graph:", output_path)
        print("[INFO] all done")

def gen_android_node_json(infer_node_path,out_info_path,input_shape=[32 * 32 * 3]):

    TargetNet = load_model('model.h5')


    layer_names = [layer.name for layer in TargetNet.layers]
    print('layer_names ',len(layer_names),layer_names[:5])

    layer_shapes = [[1]+list(layer.output_shape[1:]) for layer in TargetNet.layers]
    print('layer_shapes ',len(layer_shapes),layer_shapes[:5])

    layer_out_sizes = [reduce(lambda x, y: x * y, layer.output_shape[1:], 1) for layer in TargetNet.layers]
    print('layer_out_sizes ', len(layer_out_sizes),layer_out_sizes[:5])


    ops = open(infer_node_path,'r',encoding='utf-8').readlines()
    ops = [x.strip() for x in ops]
    print('ops ',ops[:5])


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


    layer_dependences = get_model_dependence()
    print('layer_dependences ',len(layer_dependences),layer_dependences[:5])


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

    TargetNet = load_model('model.h5')


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

    gen_freeze_graph_node_name('out/', 'frozen_VGG_node.txt')

    gen_android_node_json('frozen_VGG_node.txt','VGG_node.json')

    gen_out_size('VGG_outsize.json')
