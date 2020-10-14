# coding=utf-8
from keras import backend as K
import tensorflow as tf
import keras
from keras.models import load_model
from functools import reduce
import json
from TF2Pb import freeze_graph
from TF2Op import gen_freeze_graph_node_name,gen_out_size,gen_down_up_time,get_model_dependence
from models.Chairs.ChairsModel import ChairsNet

TargetModelName = 'Chairs'

def get_model_ops():
    f1 = open("nodes.txt", "w")
    for n in tf.get_default_graph().as_graph_def().node:
        f1.write(n.name)
        f1.write("\n")
    f1.close()

def export_model_froms( output_node_names):
    tf.train.Saver().save(K.get_session(), 'out/' + TargetModelName + '.chkp')

    freeze_graph('out/', output_graph=TargetModelName+"_model.pb",
                 output_node_names=output_node_names)


def train_model():
    TargetNet = ChairsNet().net

    TargetNet.save('model.h5')

    get_model_ops()

    return TargetNet

def gen_android_node_json(infer_node_path,out_info_path,
                          class_len=809, view_len=4, transf_param_len=12):
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

        match_ops = []
        for op in ops:

            first_name = op.split('/')[0]
            if name == first_name:

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
            bean["previousSize"] = [class_len]
            bean["previousShape"] = [[1, class_len]]
            bean['outSize'] = class_len
            bean['outShape'] = [1,class_len]
        if i == 1:
            bean['layerName'] = layer.name
            bean['opName'] = last_ops[i]
            bean["previousOpName"] = [layer.name]
            bean["previousSize"] = [view_len]
            bean["previousShape"] = [[1, view_len]]
            bean['outSize'] = view_len
            bean['outShape'] = [1,view_len]
        if i == 2:
            bean['layerName'] = layer.name
            bean['opName'] = last_ops[i]
            bean["previousOpName"] = [layer.name]
            bean["previousSize"] = [transf_param_len]
            bean["previousShape"] = [[1, transf_param_len]]
            bean['outSize'] = transf_param_len
            bean['outShape'] = [1,transf_param_len]
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


if __name__ == '__main__':

    gen_android_node_json('frozen_'+TargetModelName+'_node.txt',
                          TargetModelName+'_node.json',class_len=809, view_len=4, transf_param_len=12)

