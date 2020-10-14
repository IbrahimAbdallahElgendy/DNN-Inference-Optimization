# coding=utf-8


if __name__ == '__main__':
    lines = open('MNote5/MobileNodeComputeTime.txt','r',encoding='utf-8').readlines()
    lines = [x.strip() for x in lines]

    with open('MobileNodeComputeTime.txt','w',encoding='utf-8') as wf:
        for line in lines:
            name, time = line.split('\t')
            time = '%.4f'%(float(time) * 0.4)
            print(name, time)
            wf.write(name + '\t' + time +'\n')
