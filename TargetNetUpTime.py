# coding=utf-8

import socket
import sys
import time
if __name__ == '__main__':

    HOST = '192.168.0.177'
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


    while True:

        conn, addr = s.accept()
        t1 = time.time()
        print('Connect with ' + addr[0] + ':' + str(addr[1]))

        data = []
        buf = conn.recv(1024*1024)
        while len(buf) > 0:
            data.extend(buf)
            buf = conn.recv(1024*1024)

        t2 = time.time()
        print('data len ', len(data), ('%.4f' % ((t2 - t1)*1000)) , 'ms')

        conn.close()
        print('closed')

    s.close()


