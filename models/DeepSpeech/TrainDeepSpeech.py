# coding=utf-8
from keras import backend as K
import tensorflow as tf
import keras
from TF2Pb import freeze_graph
from TF2Op import gen_freeze_graph_node_name,gen_down_up_time,get_model_dependence
import json
from functools import reduce
from keras.models import load_model

from models.DeepSpeech.DeepSpeechModel import DeepSpeechNet

TargetModelName = 'DeepSpeech'

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

    TargetNet = DeepSpeechNet().net
    TargetNet.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                      metrics=['accuracy'])


    TargetNet.save('model.h5')

    get_model_ops()

    return TargetNet

def gen_android_node_json(infer_node_path,out_info_path,
                          sequence_len=778,input_shape=[778, 26]):

    TargetNet = load_model('model.h5')


    layer_names = [layer.name for layer in TargetNet.layers]
    print('layer_names ',len(layer_names),layer_names[:5])

    layer_shapes = [[1,sequence_len]+list(layer.output_shape[2:]) for layer in TargetNet.layers]
    print('layer_shapes ',len(layer_shapes),layer_shapes[:5])

    layer_out_sizes = [reduce(lambda x, y: x * y, layer.output_shape[2:], sequence_len) for layer in TargetNet.layers]
    print('layer_out_sizes ', len(layer_out_sizes),layer_out_sizes[:5])

    ops = open(infer_node_path,'r',encoding='utf-8').readlines()
    ops = [x.strip() for x in ops]
    print('ops ',ops[:5])


    last_ops = []
    for name in layer_names:

        match_ops = []
        for op in ops:
            first_name = op.split('/')
            if name in first_name:

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

def gen_out_size(out_info_path,sequence_len=778):

    TargetNet = load_model('model.h5')


    layer_out_sizes = [reduce(lambda x, y: x * y, layer.output_shape[2:], sequence_len) for layer in TargetNet.layers]
    print('layer_out_sizes ', len(layer_out_sizes), layer_out_sizes[:5])

    results = {'layerOutSizes':list(set([int(x) for x in layer_out_sizes]))}

    with open(out_info_path, 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':

    TargetNet = train_model()

    TargetNet.summary()


    export_model_froms(output_node_names='predictions/Reshape_1')


    gen_freeze_graph_node_name('out/', 'frozen_'+TargetModelName+'_node.txt',
                               output_node_names='predictions/Reshape_1')

    gen_android_node_json('frozen_' + TargetModelName + '_node.txt', TargetModelName + '_node.json',
                          sequence_len=256, input_shape=[256, 26])

    gen_out_size(TargetModelName+'_outsize.json',sequence_len=256)


    gen_down_up_time(TargetModelName + '_outsize.json', 'MobileNodeUploadTime.txt')
    gen_down_up_time(TargetModelName + '_outsize.json', 'MobileNodeDownloadTime.txt')

