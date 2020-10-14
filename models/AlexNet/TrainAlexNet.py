# coding=utf-8
from keras import backend as K
import tensorflow as tf
import keras
from TF2Pb import freeze_graph
from TF2Op import gen_freeze_graph_node_name,gen_android_node_json,gen_out_size,gen_down_up_time
from models.AlexNet.AlexNetModel import AlexNet

TargetModelName = 'AlexNet'

def get_model_ops():
    f1 = open("nodes.txt", "w")
    for n in tf.get_default_graph().as_graph_def().node:
        f1.write(n.name)
        f1.write("\n")
    f1.close()

def export_model_froms(model_name,  output_node_names):

    tf.train.Saver().save(K.get_session(), 'out/' + model_name + '.chkp')
    freeze_graph('out/', output_graph=TargetModelName+"_model.pb",
                 output_node_names=output_node_names)


def train_model():

    TargetNet = AlexNet().net

    TargetNet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    TargetNet.save('model.h5')

    get_model_ops()

    return TargetNet

if __name__ == '__main__':

    TargetNet = train_model()

    TargetNet.summary()

    export_model_froms(TargetModelName, output_node_names='predictions/Softmax')

    gen_freeze_graph_node_name('out/', 'frozen_'+TargetModelName+'_node.txt',output_node_names='predictions/Softmax')

    gen_android_node_json('frozen_'+TargetModelName+'_node.txt',
                          TargetModelName+'_node.json',input_shape=[224, 224, 3])

    gen_out_size(TargetModelName+'_outsize.json')

    gen_down_up_time(TargetModelName + '_outsize.json', 'MobileNodeUploadTime.txt')
    gen_down_up_time(TargetModelName + '_outsize.json', 'MobileNodeDownloadTime.txt')


