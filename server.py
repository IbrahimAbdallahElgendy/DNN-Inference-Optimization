# coding=utf-8
import socket
import sys
from keras.models import load_model
from keras import backend as K
import numpy as np
import base64
import struct
"""
在边缘设备上运行，接收部分推断结果并发送推断结果。

IP地址和端口号需要更新。
"""

# TODO: 在此处更改您的IP地址和端口.
# HOST = '192.168.123.93'
HOST = '192.168.1.101'
PORT = 8888

ret = 0

# 创建一个tcp socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('socket created')
 
# 绑定socket到主机和端口
try:
    s.bind((HOST, PORT))
except socket.error as err:
    print('Bind Failed, Error Code: ' + str(err[0]) + ', Message: ' + err[1])
    sys.exit()
 
print('Socket Bind Success!')
 
 
#listen(): 此方法设置并启动TCP监听器.
s.listen(10)
print('Socket is now listening')

# 加载 Keras model
GoogLeNet = load_model('model.h5')

# Get function to :  接收来自手机的输入 并 继续推理
layer = GoogLeNet.layers[16]
f_part2 = K.function([layer.input, K.learning_phase()], [GoogLeNet.layers[97].output])
print('Model is ready')

# 收到消息后，继续推理并将结果发送回去
while 1:
    # 监听来自其他设备的连接
    conn, addr = s.accept()
    print('Connect with ' + addr[0] + ':' + str(addr[1]))
    # 读取发送来的数据
    # buf = recvall(conn)
    buf = conn.recv(1024000)

    # 执行剩下的推理
    if len(buf) == 354128:
        decoded = base64.b64decode(buf)
        unpacked = struct.unpack('>65536f', decoded)
        cut = np.asarray(unpacked).reshape((1,16,16,256))
        cut_res = f_part2([cut, 0])[0]
        ret = np.argmax(cut_res[0])

    # 创建发送回数据的socket
    sr = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sr.connect((addr[0], 8887))
    sr.send(bytes([ret]))
    sr.close()
s.close()