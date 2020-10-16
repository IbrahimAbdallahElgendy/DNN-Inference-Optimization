# coding=utf-8
"""
 移动设备统计数据上传时间
"""
import socket
import sys
import time
if __name__ == '__main__':
    # TODO: 在此处更改您的IP地址和端口.
    HOST = '192.168.1.107'
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

    # listen(): 此方法设置并启动TCP监听器.
    s.listen(10)
    print('Socket is now listening')

    # 收消息
    while True:
        # 监听来自其他设备的连接
        conn, addr = s.accept()
        t1 = time.time()
        print('Connect with ' + addr[0] + ':' + str(addr[1]))
        # 读取发送来的数据
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


