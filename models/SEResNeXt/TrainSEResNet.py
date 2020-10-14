# coding=utf-8
from keras import backend as K
import tensorflow as tf
import keras
from keras.models import load_model
from TF2Pb import freeze_graph
import json
from functools import reduce
from TF2Op import gen_freeze_graph_node_name,gen_out_size,gen_down_up_time,get_model_dependence
from models.SEResNeXt.SEResNetXtModel import SEResNeXtNet

TargetModelName = 'SEResNeXt'

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
    TargetNet = SEResNeXtNet().net

    TargetNet.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                      metrics=['accuracy'])

    # 2.保存模型
    TargetNet.save('model.h5')

    # 3.获取所有操作名
    get_model_ops()

    return TargetNet

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
            first_name = op.split('/')  # [0]
            if name in first_name:
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

if __name__ == '__main__':
    # # 1.得到模型
    # TargetNet = train_model()
    # # 2.打印模型结构
    # TargetNet.summary()
    #
    # # 3.导出模型
    # export_model_froms(output_node_names='predictions/Softmax')
    #
    # # 4.获取和目标函数相关的操作名
    # gen_freeze_graph_node_name('out/', 'frozen_'+TargetModelName+'_node.txt',
    #                            output_node_names='predictions/Softmax')
    # 5.获取安卓端需要的计算节点信息
    gen_android_node_json('frozen_'+TargetModelName+'_node.txt',
                          TargetModelName+'_node.json',input_shape=[32, 32, 3])
    # 6.获取去重的层输出大小
    gen_out_size(TargetModelName+'_outsize.json')

    # 7.生成全0的上传下载时间文件
    gen_down_up_time(TargetModelName + '_outsize.json', 'MobileNodeUploadTime.txt')
    gen_down_up_time(TargetModelName + '_outsize.json', 'MobileNodeDownloadTime.txt')

