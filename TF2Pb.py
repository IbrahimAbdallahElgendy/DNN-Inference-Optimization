# coding=utf-8


import os, argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.framework import graph_util


def freeze_graph(model_folder, output_graph="frozen_model.pb",
                 output_node_names = "dense_5/Softmax,dense_2/Softmax,dense_4/Softmax"):

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


        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

        print("[INFO] output_graph:", output_graph)
        print("[INFO] all done")

if __name__ == '__main__':

    freeze_graph('F:\Graduation project\MasterCode\models\VGG\out','frozen_VGG.pb')

