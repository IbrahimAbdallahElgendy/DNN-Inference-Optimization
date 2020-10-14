# coding=utf-8
from keras import backend as K
import tensorflow as tf
import keras
from TF2Pb import freeze_graph
from TF2Op import gen_freeze_graph_node_name,gen_android_node_json,gen_out_size,gen_down_up_time
from models.VAE.VAEModel import VAENet

TargetModelName = 'VAE'

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
    TargetNet = VAENet().net

    # 2.保存模型
    TargetNet.save('model.h5')

    # 3.获取所有操作名
    get_model_ops()

    return TargetNet

if __name__ == '__main__':
    # 1.得到模型
    TargetNet = train_model()
    # 2.打印模型结构
    TargetNet.summary()

    # # 3.导出模型
    # export_model_froms(output_node_names='predictions/Sigmoid')
    #
    # # 4.获取和目标函数相关的操作名
    # gen_freeze_graph_node_name('out/', 'frozen_'+TargetModelName+'_node.txt',
    #                            output_node_names='predictions/Sigmoid')
    # image_size = 32
    # # 5.获取安卓端需要的计算节点信息
    # gen_android_node_json('frozen_'+TargetModelName+'_node.txt',
    #                       TargetModelName+'_node.json',input_shape=[image_size*image_size])
    # # 6.获取去重的层输出大小
    # gen_out_size(TargetModelName+'_outsize.json')
    #
    # # 7.生成全0的上传下载时间文件
    # gen_down_up_time(TargetModelName + '_outsize.json', 'MobileNodeUploadTime.txt')
    # gen_down_up_time(TargetModelName + '_outsize.json', 'MobileNodeDownloadTime.txt')

