# coding=utf-8
from keras import backend as K
import tensorflow as tf
import keras
from TF2Pb import freeze_graph
from TF2Op import gen_freeze_graph_node_name,gen_down_up_time,get_model_dependence
from models.seq2seq.Seq2SeqModel import Seq2SeqNet
import json
from functools import reduce
from keras.models import load_model

TargetModelName = 'Seq2Seq'

def get_model_ops():
    ''' nodes.txt 里写入模型的节点名字 '''
    f1 = open("nodes.txt", "w")
    for n in tf.get_default_graph().as_graph_def().node:
        f1.write(n.name)
        f1.write("\n")
    f1.close()

def export_model_froms( output_node_names):
    '''
    将 session 写到文件 out/googlenet.chkp.***
    .meta文件保存了当前图结构；.index文件保存了当前参数名；.data文件保存了当前参数值。
    '''
    tf.train.Saver().save(K.get_session(), 'out/' + TargetModelName + '.chkp')

    freeze_graph('out/', output_graph=TargetModelName+"_model.pb",
                 output_node_names=output_node_names)


def train_model():
    # 1.搭建模型结构
    TargetNet = Seq2SeqNet().net

    # 2.保存模型
    TargetNet.save('model.h5')

    # 3.获取所有操作名
    get_model_ops()

    return TargetNet

def gen_android_node_json(infer_node_path,out_info_path,
                          seq_len=20,model_input_shape=[20,256]):
    # 加载模型
    TargetNet = load_model('model.h5')

    # 取得层名
    layer_names = [layer.name for layer in TargetNet.layers]
    print('layer_names ',len(layer_names),layer_names)
    # 层shape
    # layer_shapes = [[1,seq_len]+list(layer.output_shape[2:]) for layer in TargetNet.layers]
    layer_shapes = []
    for layer in TargetNet.layers:
        if type(layer.output_shape) == list:
            print('multi out layer ',layer.name,layer.output_shape)
            tmp_shape = []
            for shape in layer.output_shape:
                tmp_shape.append([1] + list(shape[1:]))
            layer_shapes.append(tmp_shape)
        else:
            layer_shapes.append([1,seq_len] + list(layer.output_shape[2:]) )
    print('layer_shapes ',len(layer_shapes),layer_shapes)
    # 层大小
    # layer_out_sizes = [reduce(lambda x, y: x * y, layer.output_shape[1:], 1) for layer in TargetNet.layers]
    layer_out_sizes = []
    for shape in layer_shapes:
        if type(shape[0]) == list:
            sum_size = 0
            for tmp_shape in shape:
                sum_size += reduce(lambda x, y: x * y, tmp_shape, 1)
            layer_out_sizes.append(sum_size)
        else:
            layer_out_sizes.append(reduce(lambda x, y: x * y, shape, 1))
    print('layer_out_sizes ', len(layer_out_sizes),layer_out_sizes)

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
            first_name = op.split('/')#[0]
            if name in first_name:
                # print('first_name', first_name)
                match_ops.append(op)
        last_ops.append(match_ops[-1])

    print('len last_ops ',len(last_ops),last_ops)

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
            bean["previousSize"] = model_input_shape
            bean["previousShape"] = [[1, *model_input_shape]]
            bean['outSize'] = int(reduce(lambda x, y: x * y, model_input_shape, 1))
            bean['outShape'] = [1, *model_input_shape]
        elif 'input' in layer.name:
            bean['layerName'] = layer.name
            bean['opName'] = last_ops[i]
            bean["previousOpName"] = [layer.name]
            bean["previousSize"] = model_input_shape
            bean["previousShape"] = [[1, *model_input_shape]]
            bean['outSize'] = int(reduce(lambda x, y: x * y, model_input_shape, 1))
            bean['outShape'] = [1, *model_input_shape]
        else:
            bean['layerName'] = layer.name
            bean['opName'] = last_ops[i]
            bean["previousOpName"] = [last_ops[name2index[x[0]]] for x in layer_dependences[i]]
            bean["previousSize"] = [layer_out_sizes[name2index[x[0]]] for x in layer_dependences[i]]
            bean["previousSize"] = [int(x) for x in bean["previousSize"]]

            # print('layer_dependences[i]',layer_dependences[i])
            # print([ layer_shapes[name2index[x[0]]] for x in layer_dependences[i]])
            bean["previousShape"] = []
            for pre_name, _, shap_index in layer_dependences[i]:
                tmp_shape = layer_shapes[name2index[pre_name]]
                print(pre_name, shap_index, tmp_shape)
                if type(tmp_shape[0]) == list:
                    bean["previousShape"].append(tmp_shape[shap_index])
                else:
                    bean["previousShape"].append(tmp_shape)
            # print('bean["previousShape"]',bean["previousShape"])

            bean["previousShape"] = [[int(a) for a in x] for x in bean["previousShape"]]
            bean['outSize'] = int(layer_out_sizes[i])
            if type(layer_shapes[i][0]) == list:
                bean['outShape'] = [[int(x) for x in l] for l in layer_shapes[i]]
            else:
                bean['outShape'] = [int(x) for x in layer_shapes[i]]

        print(bean)
        results.append(bean)

    with open(out_info_path, 'w') as f:
        json.dump(results, f)

def gen_out_size(out_info_path,seq_len=20):
    # 加载模型
    TargetNet = load_model('model.h5')

    # 层shape
    layer_shapes = []
    for layer in TargetNet.layers:
        if type(layer.output_shape) == list:
            print('multi out layer ', layer.name, layer.output_shape)
            tmp_shape = []
            for shape in layer.output_shape:
                tmp_shape.append([1] + list(shape[1:]))
            layer_shapes.append(tmp_shape)
        else:
            layer_shapes.append([1, seq_len] + list(layer.output_shape[2:]))
    print('layer_shapes ', len(layer_shapes), layer_shapes)
    # 层大小
    layer_out_sizes = []
    for shape in layer_shapes:
        if type(shape[0]) == list:
            sum_size = 0
            for tmp_shape in shape:
                sum_size += reduce(lambda x, y: x * y, tmp_shape, 1)
            layer_out_sizes.append(sum_size)
        else:
            layer_out_sizes.append(reduce(lambda x, y: x * y, shape, 1))
    print('layer_out_sizes ', len(layer_out_sizes), layer_out_sizes)

    results = {'layerOutSizes':list(set([int(x) for x in layer_out_sizes]))}

    with open(out_info_path, 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    # 1.得到模型
    TargetNet = train_model()
    # 2.打印模型结构
    TargetNet.summary()

    # 3.导出模型
    export_model_froms(output_node_names='predictions/truediv')

    # 4.获取和目标函数相关的操作名
    gen_freeze_graph_node_name('out/', 'frozen_'+TargetModelName+'_node.txt',
                               output_node_names='predictions/truediv')

    # 5.获取安卓端需要的计算节点信息
    gen_android_node_json('frozen_'+TargetModelName+'_node.txt',
                          TargetModelName+'_node.json',seq_len=20,model_input_shape=[20])
    # 6.获取去重的层输出大小
    gen_out_size(TargetModelName+'_outsize.json',seq_len=20)

    # 7.生成全0的上传下载时间文件
    gen_down_up_time(TargetModelName + '_outsize.json', 'MobileNodeUploadTime.txt')
    gen_down_up_time(TargetModelName + '_outsize.json', 'MobileNodeDownloadTime.txt')

