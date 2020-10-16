# coding=utf-8
"""
Tensorflow graph 冷冻机
转换Tensorflow训练模型为.pb

Code adapted from:
https://gist.github.com/morgangiraud/249505f540a5e53a48b0c1a869d370bf#file-medium-tffreeze-1-py
"""

import os, argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.framework import graph_util


def freeze_graph(model_folder, output_graph="frozen_model.pb",
                 output_node_names = "dense_5/Softmax,dense_2/Softmax,dense_4/Softmax"):
    # 我们获取检查点全路径
    try:
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path
        print("[INFO] input_checkpoint:", input_checkpoint)
    except:
        input_checkpoint = model_folder
        print("[INFO] Model folder", model_folder)

    # # 在导出graph之前，我们需要精确确定output node
    # # 这就是TF决定他必须保留Graph的哪一部分以及可以丢弃哪一部分的方式
    # output_node_names = "dense_5/Softmax"  # NOTE: 改变这里!!!

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

        # 最后，我们序列化并将输出图转储到文件系统
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

        print("[INFO] output_graph:", output_graph)
        print("[INFO] all done")

if __name__ == '__main__':

    freeze_graph('out/','frozen_googlenet.pb')

