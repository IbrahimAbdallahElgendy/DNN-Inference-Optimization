# coding=utf-8
import networkx as nx
from DNNTopologyHandler import construct_DNN_topology,construct_multi_input_DNN_topology

def DADS_partition(nodes,S,T,INI_MAX=100000):

    DG = nx.DiGraph()
    DG.add_nodes_from(["e","m"])# m 代表mobile ，e代表 edge

    """ 添加计算边 """
    # 开始节点必须割tm节点
    DG.add_edge('e', S.layer_name, capacity=INI_MAX) # 一定不会割到这条边
    DG.add_edge( S.layer_name,'m', capacity=0)
    ##### multi input 之前 ######
    assert len(S.next_nodes) == 1
    queue = []
    used_flag = []
    queue.append(S.next_nodes[0])
    used_flag.append(S.next_nodes[0])
    ##### multi input 之前 ######
    while len(queue) > 0:
        current_name = queue[0]
        queue.remove(current_name)
        current_node = nodes[current_name]
        print(current_node.layer_name, 'in add compute time edge')

        DG.add_node(current_name)
        DG.add_edge('e', current_name, capacity=current_node.edge_ct)
        DG.add_edge(current_name, 'm', capacity=current_node.mobile_ct)

        if current_node.next_nodes:
            for next_name in current_node.next_nodes:
                if next_name not in used_flag and next_name != "T":
                    queue.append(next_name)
                    used_flag.append(next_name)


    queue = []
    used_flag = []
    # queue.append(S.next_nodes[0])
    # used_flag.append(S.next_nodes[0])
    queue.append(S.layer_name)
    used_flag.append(S.layer_name)
    while len(queue) > 0:
        current_name = queue[0]
        queue.remove(current_name)
        current_node = nodes[current_name]
        print(current_node.layer_name, 'in add trans time edge')

        if len(current_node.next_nodes) == 1:
            # 只有一个后继
            next_name = current_node.next_nodes[0]
            if next_name == 'T':
                continue
            DG.add_edge(current_name, next_name, capacity=current_node.upload_tt)

        else:
            # 多个后继
            aux_name = current_name + '_'
            DG.add_edge(current_name, aux_name, capacity=current_node.upload_tt)
            for next_name in current_node.next_nodes:# 要加上 => 表示不能切割这个位置
                DG.add_edge(aux_name,next_name,capacity=INI_MAX)

        for next_name in current_node.next_nodes:
            if next_name not in used_flag and next_name != "T":
                queue.append(next_name)
                used_flag.append(next_name)

    # 结束节点必须割te节点
    DG.add_edge(T.layer_name, 'm', capacity=INI_MAX)  # 一定不会割到这条边 => 必须结束在边缘
    '''输出下载的总时间'''
    download_result_tt = 0
    for pre_name in T.previous_nodes:
        download_result_tt += nodes[pre_name].download_tt
        DG.add_edge(pre_name, T.layer_name, capacity=INI_MAX) # 不让切，输出节点的前驱节点肯定是在边缘执行
    DG.add_edge('e', T.layer_name, capacity=download_result_tt)

    # ''' 打印图结构'''
    # for dg_node in DG.nodes:
    #     print("+"*10,dg_node,"+"*10)
    #     for nbr in list(DG.successors(dg_node)):
    #         print(nbr,DG[dg_node][nbr])


    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(36, 36))
    # nx.draw(DG, with_labels=True, font_weight='bold')
    # plt.show()

    cut_value, partition = nx.minimum_cut(DG, 'e', 'm')
    print('cut_value ',cut_value)
    reachable, non_reachable = partition
    print('len reachable', len(reachable))
    print('reachable', reachable)
    print('len non_reachable', len(non_reachable))
    print('non_reachable', non_reachable)

    cutset = set()
    for u, nbrs in ((n, DG[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)
    print('sorted(cutset)........')
    print(sorted(cutset))
    compute_in_edge = []
    compute_in_mobile = []
    for cut_edge in cutset:
        start ,end = cut_edge
        if start == 'e':
            compute_in_edge.append(end)
        if end == 'm':
            compute_in_mobile.append(start)
    print('compute_in_mobile ',compute_in_mobile )
    print('compute_in_edge ',compute_in_edge)


    '''统计在移动设备计算时间'''
    accumulate_mobile_ct = 0
    for in_mobile_name in reachable:
        if in_mobile_name == 'e' or in_mobile_name[-1] == '_':
            continue
        accumulate_mobile_ct += nodes[in_mobile_name].mobile_ct
    '''统计在边缘计算时间'''
    accumulate_edge_ct = 0
    for in_edge_name in non_reachable:
        if in_edge_name == 'm' or in_edge_name[-1] == '_' or in_edge_name == T.layer_name:
            continue
        accumulate_edge_ct += nodes[in_edge_name].edge_ct

    '''统计数据传输时间'''
    accumulate_upload_tt = 0
    cut_place_nodes = []
    for in_mobile_name in reachable:
        if in_mobile_name == 'e' or in_mobile_name[-1] == '_' \
                or in_mobile_name == T.layer_name:
            continue
        need_up_flag = False
        if len(nodes[in_mobile_name].next_nodes) == 1:
            #不是分支结构，数据传输看后继
            next_name = nodes[in_mobile_name].next_nodes[0]
            if next_name in non_reachable:
                need_up_flag = True
        else:
            # 分支结构，看辅助节点
            aux_name = in_mobile_name + '_'
            if aux_name in non_reachable:
                need_up_flag = True

        if need_up_flag:
            cut_place_nodes.append(in_mobile_name)
            print(in_mobile_name,'need up to edge...')
            accumulate_upload_tt += nodes[in_mobile_name].upload_tt

    '''合并端到端计算延迟'''
    end2end_infer_time = accumulate_mobile_ct + accumulate_upload_tt + \
                         accumulate_edge_ct + download_result_tt
    print('end2end_infer_time: %s = %s + %s + %s + %s' % (
        'end2end_infer_time', 'accumulate_mobile_ct',
        'accumulate_upload_tt', 'accumulate_edge_ct', 'download_result_tt'))
    print('end2end_infer_time: %.4f = %.4f + %.4f + %.4f + %.4f'%(
        end2end_infer_time,accumulate_mobile_ct,accumulate_upload_tt,accumulate_edge_ct,download_result_tt
    ))

    return cut_place_nodes,end2end_infer_time

if __name__ == '__main__':
    # VGG
    nodes, S = construct_DNN_topology(Model_Path="models/VGG/", main_target_name='predictions',mode='WIFI')

    """分区算法执行"""
    print('+'*100)
    print('DADS_partition Algo run...')
    cut_place_nodes, end2end_infer_time = DADS_partition(nodes, S,nodes['T'])
    print('+' * 100)
    print('need up to edge',cut_place_nodes,'end2end_infer_time',end2end_infer_time)
