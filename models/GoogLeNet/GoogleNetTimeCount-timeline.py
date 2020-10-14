# coding=utf-8

from keras import optimizers
import numpy as np
import tensorflow as tf
import os
import json
import time
from tensorflow.python.client import timeline

from models.GoogLeNet.GoogleNetModel import GoogLeNetModel

TIMELINE_DIR = 'out/timeline'
FOCUS_OP_SET = set(['Conv2D','BiasAdd','Relu','MaxPool','Mean','MatMul','Softmax',
                    'Mul','Sub','FusedBatchNorm','Merge','Switch','Identity','VariableV2',
                    'AvgPool','Add','ConcatV2','Const','Rsqrt',
                    ])

""" 导入模型-法1 """
# 创建模型
GoogLeNet = GoogLeNetModel().net
# 定义optimizer并编译模型
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# loss_weights - weighting losses of main classifier and auxiliary classifiers
# GoogLeNet.compile(optimizer=optimizer, loss='categorical_crossentropy',
#                   metrics = ['accuracy'], loss_weights=[1., 0.3, 0.3])
""" 导入模型-法2 """
# # 加载 Keras model
# GoogLeNet = load_model('model.h5')

# 打印模型结构
GoogLeNet.summary()

# 模型层
for layer in GoogLeNet.layers:
    print(layer)

print('len ',len(GoogLeNet.layers))

# 获得每层的执行时间
# step 1： 定义 run_options 和 run_metadata, 用于保存op的属性
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
# step 2： 将  run_options 和 run_metadata 添加至 model的 compile方法
GoogLeNet.compile(loss='categorical_crossentropy',metrics = ['accuracy'],
                  loss_weights=[1., 0.3, 0.3],optimizer=optimizer,
                  options=run_options,run_metadata=run_metadata)
test_x = np.random.rand(32, 32, 3)
begin = time.time()
GoogLeNet.predict(np.array([test_x]))
end = time.time()
# step 3: 在 timeline.json中保存 run_metadate
tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format()
with open(os.path.join(TIMELINE_DIR, 'googlenet-timeline.json'), 'w') as f:
    f.write(ctf)

print('infer time ',end-begin)

def get_node_runtime_list(timeline_path):
    with open(timeline_path, 'r') as load_f:
        runtimes = json.load(load_f)
        traceEvents = runtimes['traceEvents']
        print('len traceEvents ',len(traceEvents))
        OpNames = set()
        useful_ops = []
        # 找到需要的节点 => FOCUS_OP_SET中关注的op节点
        for node in traceEvents:
            if 'cat' in node and node['cat'] == 'Op':
                print(node['name'], node['args']['name'], node['dur'])
                OpNames.add(node['name'])
                if node['name'] in FOCUS_OP_SET:
                    useful_ops.append((node['name'],node['args']['name'], node['dur']))

        print('len OpNames ',len(OpNames))
        print(OpNames)

        print('len useful_ops ',len(useful_ops))
        print(useful_ops)

        # 合并同类op节点
        merge_useful_ops = []
        last_op_name = ''
        for op in useful_ops:
            op_name = op[1].split('/')[0]#node['args']['name']
            if op_name == last_op_name:
                merge_useful_ops[-1][1] += op[2]
            else:
                merge_useful_ops.append([op_name,op[2]])
            last_op_name = op_name

        print('len merge_useful_ops ',len(merge_useful_ops))
        print(merge_useful_ops)
        return merge_useful_ops

# 得到每个操作执行时间的列表
timeline_path = os.path.join(TIMELINE_DIR, 'googlenet-timeline.json')
node_runtime_list = get_node_runtime_list(timeline_path)
print(node_runtime_list)

times = [x[1] for x in node_runtime_list]
print(times)
print(len(times))
print(np.array(times).sum())
