import tensorflow as tf
import os
def wirte_graph_log(meta_file,log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    g = tf.Graph()
    with g.as_default() as g:
        tf.train.import_meta_graph(meta_file)
    with tf.Session(graph=g) as sess:
        tf.summary.FileWriter(logdir=log_dir,graph=g)
if __name__ == '__main__':
   wirte_graph_log('\MasterCode\models\ResNet\out\ResNet.chkp.meta','./log/')