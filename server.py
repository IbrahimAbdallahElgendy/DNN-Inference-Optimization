# coding=utf-8
import socket
import sys
from keras.models import load_model
from keras import backend as K
import numpy as np
import base64
import struct



HOST = '192.168.1.100'
PORT = 8000

ret = 0


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('socket created')
 

try:
    s.bind((HOST, PORT))
except socket.error as err:
    print('Bind Failed, Error Code: ' + str(err[0]) + ', Message: ' + err[1])
    sys.exit()
 
print('Socket Bind Success!')
 
 

s.listen(10)
print('Socket is now listening')


VGG = load_model('F:\Graduation project\MasterCode\models\VGG\model.h5')


layer = VGG.layers[0]
# print(layer)
# print(VGG.layers[97])
f_part2 = K.function([layer.input, K.learning_phase()], [VGG.layers[7].output])
print('Model is ready')


while 1:

    conn, addr = s.accept()
    print('Connect with ' + addr[0] + ':' + str(addr[1]))

    # buf = recvall(conn)
    buf = conn.recv(1024000)


    if len(buf) == 354128:
        decoded = base64.b64decode(buf)
        unpacked = struct.unpack('>65536f', decoded)
        cut = np.asarray(unpacked).reshape((1,16,16,256))
        cut_res = f_part2([cut, 0])[0]
        ret = np.argmax(cut_res[0])


    sr = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sr.connect((addr[0], 8887))
    sr.send(bytes([ret]))
    sr.close()
s.close()