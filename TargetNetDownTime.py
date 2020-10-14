# coding=utf-8

import socket
import json
import time

def run_download_time(outsize_path):

    run_num = 200
    # outsize
    f = open(outsize_path, encoding='utf-8')
    layerOutSizes = json.load(f)['layerOutSizes']
    print(layerOutSizes)

    # Server IP
    remote_ip = '192.168.0.103'
    # Server port
    remote_port = 8000

    layerOutSizes = sorted(layerOutSizes)
    print('sorted layerOutSizes ',layerOutSizes)
    for size in layerOutSizes:
        print('size', size)
        sum_down_time = 0
        for i in range(run_num):
            start = time.time()

            tcpClientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            serAddr = (remote_ip, remote_port)
            tcpClientSocket.connect(serAddr)

            bytearr = bytes([2] * size * 4)
            tcpClientSocket.send(bytearr)
            tcpClientSocket.close()

            end = time.time()
            down_time = (end - start)*1000
            sum_down_time += down_time
            print('len ', len(bytearr), 'time ', ('%.4f' % (down_time)))

            # time.sleep(0.050) #50ms

        mean_down_time = sum_down_time / run_num
        print('mean_down_time', '%.4f' % mean_down_time)
        time.sleep(5)

if __name__ == '__main__':
    # GoogleNet
    # run_download_time('models/GoogLeNet/googlenet_outsize.json')
    # VGG
    run_download_time('F:\Graduation project\MasterCode\models\VGG\VGG_outsize.json')








