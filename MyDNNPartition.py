# coding=utf-8
import copy
from DNNTopologyHandler import construct_DNN_topology,Node,construct_multi_input_DNN_topology
"""
线性拓扑中DP的状态转移
"""
def dp_state_move_to_next(nodes,current_node,dp_mobile,path_mobile,dp_edge,path_edge):

    assert len(current_node.previous_nodes) == 1  # 线性拓扑的算法，除S节点外每个节点有且仅有一个前驱
    """
    一.dp 状态转移方程：
        1. dp_mobile[i+1] = min { dp_mobile[i] + mobile_ct[i+1] , dp_edge[i] + download_tt[i] + mobile_ct[i+1]}
        2. dp_edge[i+1] = min { dp_edge[i] + edge_ct[i+1] , dp_mobile[i] + upload_tt[i] + edge_ct[i+1]}
    """
    last_node_name = current_node.previous_nodes[0]
    current_node_name = current_node.layer_name
    s_mobile = min(dp_mobile[last_node_name] + nodes[current_node_name].mobile_ct,
                   dp_edge[last_node_name] + nodes[last_node_name].download_tt + nodes[current_node_name].mobile_ct)
    s_edge = min(dp_edge[last_node_name] + nodes[current_node_name].edge_ct,
                 dp_mobile[last_node_name] + nodes[last_node_name].upload_tt + nodes[current_node_name].edge_ct)
    dp_mobile[current_node_name] = s_mobile
    dp_edge[current_node_name] = s_edge
    """debug info"""
    print('=' * 10, current_node_name, '=' * 10)
    print("s_mobile: %.4f  = min(%.4f + %.4f,%.4f + %.4f + %.4f)"
          % (s_mobile, dp_mobile[last_node_name], nodes[current_node_name].mobile_ct,
             dp_edge[last_node_name], nodes[last_node_name].download_tt, nodes[current_node_name].mobile_ct))
    print("s_edge: %.4f = min(%.4f + %.4f,%.4f + %.4f + %.4f)"
          % (s_edge, dp_edge[last_node_name], nodes[current_node_name].edge_ct,
             dp_mobile[last_node_name], nodes[last_node_name].upload_tt, nodes[current_node_name].edge_ct))
    """
    二.记录路径，便于回溯
    """
    if (dp_mobile[last_node_name] + nodes[current_node_name].mobile_ct <
            dp_edge[last_node_name] + nodes[last_node_name].download_tt + nodes[current_node_name].mobile_ct):
        path_mobile[current_node_name] = (last_node_name, 'mobile')
    else:
        path_mobile[current_node_name] = (last_node_name, 'edge')
    if (dp_edge[last_node_name] + nodes[current_node_name].edge_ct <
            dp_mobile[last_node_name] + nodes[last_node_name].upload_tt + nodes[current_node_name].edge_ct):
        path_edge[current_node_name] = (last_node_name, 'edge')
    else:
        path_edge[current_node_name] = (last_node_name, 'mobile')

"""
线性拓扑DNN的分区算法
"""
def linear_partition_algorithm(nodes,S,T,input_in_mobile=True):
    '''
    input_in_mobile: 表示原始输入数据是否在mobile
    d_mobile: 状态结束时在移动设备执行的用时
    d_edge: 状态结束时在边缘设备执行的用时
    '''
    # dp = [{'d_mobile':0,'d_edge':0 } ] # 第一个元素代表S开始的时候d_mobile和d_edge的值
    dp_mobile = {}
    dp_edge = {}
    path_mobile = {}
    path_edge = {}
    if input_in_mobile:
        dp_mobile[S.layer_name] = 0
        dp_edge[S.layer_name] = S.upload_tt
    else:
        dp_mobile[S.layer_name] = S.download_tt
        dp_edge[S.layer_name] = 0


    assert len(S.next_nodes) == 1  # 线性拓扑的算法，除T节点外每个节点有且仅有一个后继
    current_node = nodes[S.next_nodes[0]]
    # while current_node.layer_name != "T":
    while current_node.layer_name != T.layer_name:
        print(current_node.layer_name,'in linear_partition_algorithm ')
        assert len(current_node.previous_nodes) == 1  # 线性拓扑的算法，除S节点外每个节点有且仅有一个前驱
        """ DP状态方程转移 & 路径回溯"""
        dp_state_move_to_next(nodes,current_node, dp_mobile, path_mobile, dp_edge, path_edge)
        # 下一个节点
        assert len(current_node.next_nodes) == 1  # 线性拓扑的算法，除T节点外每个节点有且仅有一个后继
        current_node = nodes[ current_node.next_nodes[0] ]

    # current_node 现在应该是T节点
    assert len(current_node.previous_nodes) == 1
    last_node_name = current_node.previous_nodes[0]
    # 最终的结果提供给边缘
    T_end_in_edge = min(dp_edge[last_node_name],dp_mobile[last_node_name] + nodes[last_node_name].upload_tt)
    # 最终的结果提供给mobile
    T_end_in_mobile = min(dp_mobile[last_node_name],dp_edge[last_node_name] + nodes[last_node_name].download_tt)
    return T_end_in_edge,path_edge,T_end_in_mobile,path_mobile

def find_branch_end_node(nodes, branch_begin_node):
    branch_end_candidates = []
    # 记录遇到分支结束候选节点之前遇到的分支开始节点
    branch_begin_names = []
    begin_names_flag = []
    branch_begin_names.append(branch_begin_node.layer_name)
    begin_names_flag.append(branch_begin_node.layer_name)
    while len(branch_begin_names) > 0:
        begin_name = branch_begin_names[0]
        branch_begin_names.remove(begin_name)
        begin_node = nodes[begin_name]
        # print('begin_name',begin_name)

        begin_next_node_names = begin_node.next_nodes
        for next_name in begin_next_node_names:
            next_node = nodes[next_name]
            if next_node.next_nodes == None:
                print('find T node ',next_name,'return as branch end node')
                branch_end_node_name = next_name
                print('branch_end_node_name', branch_end_node_name, 'branch_end_index', next_node.index)
            if next_node.previous_nodes == None:
                print('next_node.previous_nodes == None','branch_begin_node.layer_name',branch_begin_node.layer_name,'next_node.layer_name',next_node.layer_name)
                for k,v in nodes.items():
                    print(k,v)
            # 既没有多个后继，也没有多个前驱，移动知道有多个后继
            while len(next_node.previous_nodes) == 1 and len(next_node.next_nodes) == 1:
                # print('next_name',next_name,'move next')
                next_name = next_node.next_nodes[0]
                next_node = nodes[next_name]

                if next_node.next_nodes == None:
                    print('find T node ', next_name, 'return as branch end node')
                    branch_end_node_name = next_name
                    print('branch_end_node_name', branch_end_node_name, 'branch_end_index', next_node.index)

            # 节点有多个后继，找到这条分支的第一个分支结束节点，该条路遍历结束
            if len(next_node.previous_nodes) > 1:
                # print('next_name', next_name, 'more previous end_candidates')
                if next_name not in branch_end_candidates:
                    branch_end_candidates.append(next_name)
                    # print('branch end candidate find ',next_name)
            # 一个新的分支开始节点，加入branch_begin_names
            elif len(next_node.next_nodes) > 1:
                # print('next_name', next_name, 'more next branch_begin_names')
                if next_name not in begin_names_flag:
                    branch_begin_names.append(next_name)
                    begin_names_flag.append(next_name)
                    # print('median branch begin find ', next_name)
            else:
                print('error while end...')

    print('branch_end_candidates ',branch_end_candidates)
    branch_end_index = 0
    branch_end_node_name = None
    for end_name in branch_end_candidates:
        end_node = nodes[end_name]
        if end_node.index > branch_end_index:
            branch_end_index = end_node.index
            branch_end_node_name = end_name
    print('branch_end_node_name',branch_end_node_name,'branch_end_index',branch_end_index)
    return branch_end_node_name

"""
从线性分支节点状态转移到分支结束节点
"""
def dp_state_move_to_branch_end(nodes,branch_begin_node,
                                dp_mobile, path_mobile, dp_edge, path_edge):

    print('*'*15,'move to branch end ,branch_begin_node ',branch_begin_node.layer_name,'*'*15)

    branch_begin_node_name = branch_begin_node.layer_name
    # 一. 找到最远的分支结束节点
    branch_end_node_name = find_branch_end_node(nodes, branch_begin_node)
    branch_results = []
    for next_name in branch_begin_node.next_nodes:
        print('handle branch first name ',next_name)

        new_nodes = {}
        que = []
        names_flag = []
        end_previous_name = []

        if next_name == branch_end_node_name:# 该条线路上没有任何节点
            # 新的开始节点
            new_S = Node(layer_name='S',
                          next_nodes=[next_name], previous_nodes=None,
                          edge_ct=branch_begin_node.edge_ct, mobile_ct=branch_begin_node.mobile_ct,
                          upload_tt=branch_begin_node.upload_tt, download_tt=branch_begin_node.download_tt,
                          out_size=branch_begin_node.out_size, out_shape=branch_begin_node.out_shape, index=branch_begin_node.index)
            new_nodes['S'] = new_S
            end_previous_name.append('S')

        else:
            # 新的开始节点
            new_S = copy.deepcopy(nodes[next_name])
            new_S.previous_nodes = None
            new_nodes[next_name] = new_S

            que.extend(nodes[next_name].next_nodes)
            names_flag.extend(nodes[next_name].next_nodes)

            if branch_end_node_name in nodes[next_name].next_nodes:
                que.remove(branch_end_node_name)
                names_flag.remove(branch_end_node_name)
                end_previous_name.append(next_name)

        while len(que) > 0:
            node_name = que[0]
            que.remove(node_name)

            node = nodes[node_name]
            new_nodes[node_name] = node

            if branch_end_node_name in node.next_nodes:
                # assert len(node.next_nodes) == 1
                end_previous_name.append(node_name)
            else:
                for name in node.next_nodes:
                    if name not in names_flag:
                        que.append(name)
                        names_flag.append(name)

        new_T = Node(layer_name=branch_end_node_name,
                     next_nodes=None, previous_nodes=end_previous_name,
                     edge_ct=0, mobile_ct=0,
                     upload_tt=0, download_tt=0,
                     out_size=0, out_shape=None,index=nodes[branch_end_node_name].index)
        new_nodes[branch_end_node_name] = new_T
        for _, n in new_nodes.items():
            print(n)

        T_me, sub_path_edge, T_mm, sub_path_mobile = nonlinear_partition_algorithm(new_nodes,new_S, new_T,input_in_mobile=True)
        T_me_path = path_track_back(sub_path_edge, sub_path_mobile, new_nodes,end_mobile=False,end_name=branch_end_node_name,print_info=True)
        T_mm_path = path_track_back(sub_path_edge, sub_path_mobile, new_nodes,end_mobile=True,end_name=branch_end_node_name,print_info=False)

        T_ee,sub_path_edge, T_em, sub_path_mobile = nonlinear_partition_algorithm(new_nodes,new_S, new_T,input_in_mobile=False)
        T_ee_path = path_track_back(sub_path_edge, sub_path_mobile, new_nodes, end_mobile=False, end_name=branch_end_node_name,print_info=False)
        T_em_path = path_track_back(sub_path_edge, sub_path_mobile, new_nodes, end_mobile=True, end_name=branch_end_node_name,print_info=False)

        branch_results.append({'next_node_name': next_name,
                               'T_me': T_me, 'T_me_path': T_me_path,
                               'T_mm': T_mm, 'T_mm_path': T_mm_path,
                               'T_ee': T_ee, 'T_ee_path': T_ee_path,
                               'T_em': T_em, 'T_em_path': T_em_path})

    # 二.处理分支结束节点 => 转移过去有6种可能情况
    # 1. 分支源点和分支汇点在同侧执行
    ## a.后继节点1,2,…k都和分支源点0、分支汇点k+1同侧执行（例如节点0、k+1都在边缘侧执行，而中间节点1,2,…k也在边缘侧执行
    ## b.后继节点1,2,…k都和分支源点0、分支汇点k+1不同侧执行（例如节点0、k+1都在边缘侧执行，而后继节点1,2,…k也在端侧执行，则分支的总延迟
    ## c.后继节点1,2,…k部分和分支源点0、分支汇点k+1同侧执行，部分不同侧（例如节点0、k+1都在边缘侧执行
    # 2. 分支源点和分支汇点在不同侧执行（例如，分支源点在边缘设备上执行，分支汇点在端设备执行）
    ## a.后继节点1,2,…k都和分支源点0同侧执行（例如节点0在边缘侧执行，节点k+1在端侧执行，而后继节点1,2,…k也在边缘侧执行，则分支的总延迟
    ## b.后继节点1,2,…k都和分支汇点k+1同侧执行（例如节点0在边缘侧执行，节点k+1在端侧执行，而后继节点1,2,…k也在端侧执行，则分支的总延迟
    ## c.后继节点1,2,…k部分和分支源点0、分支汇点k+1同侧执行，部分不同侧（例如节点0在边缘侧执行，节点k+1在端侧执行，则分支的总延迟
    """
    一.dp 状态转移方程：
    1.dp_mobile[branch_end_node_name] = min{
                                            dp_mobile[branch_begin_node_name] + {
                                            1.a  sum{mobile_ct[next_node_name] + T_mm[next_node_name]} 
                                            1.b  upload_tt[branch_begin_node_name] + sum{ edge_ct[next_node_name] + T_em[next_node_name]}
                                            1.c  upload_tt[branch_begin_node_name] + sum{ min( mobile_ct[next_node_name] + T_mm[next_node_name] ,edge_ct[next_node_name] + T_em[next_node_name])}
                                            } +  mobile_ct[branch_end_node_name]
                                            
                                            dp_edge[branch_begin_node_name]+ {
                                            2.a  sum{edge_ct[next_node_name] + T_em[next_node_name]}
                                            2.b  download_tt[branch_begin_node_name] + sum{mobile_ct[next_node_name] + T_mm[next_node_name]}
                                            2.c  download_tt[branch_begin_node_name] + sum{min(mobile_ct[next_node_name] + T_mm[next_node_name],edge_ct[next_node_name] + T_em[next_node_name])} 
                                            }+ mobile_ct[branch_end_node_name]
                                            }
    2.dp_edge[branch_end_node_name] = min{
                                            dp_edge[branch_begin_node_name] + {
                                            1.a. sum{ edge_ct[next_node_name] + T_ee[next_node_name]}
                                            1.b  download_tt[branch_begin_node_name] + sum{ mobile_ct[next_node_name] + T_me[next_node_name]}
                                            1.c  download_tt[branch_begin_node_name] + sum{ min(edge_ct[next_node_name] + T_ee[next_node_name] ,mobile_ct[next_node_name] + T_me[next_node_name])} 
                                            } + edge_ct[branch_end_node_name]
                                            
                                            dp_mobile[branch_begin_node_name] + {
                                            2.a  sum{mobile_ct[next_node_name] + T_me[next_node_name]}
                                            2.b  upload_tt[branch_begin_node_name] + sum{edge_ct[next_node_name] + T_ee[next_node_name]}
                                            2.c  upload_tt[branch_begin_node_name] + sum{ min(edge_ct[next_node_name] + T_ee[next_node_name] ,mobile_ct[next_node_name] + T_me[next_node_name])} 
                                            } + edge_ct[branch_end_node_name]
                                            }
    """

    ''' 1. 分支结束节点在端设备执行 '''
    print('=' * 10, 'branch end dp ',branch_end_node_name, '=' * 10)

    # 1.1 分支开始节点在端设备
    mobile_begin_min_value = 0
    mobile_begin_min_path = []
    # a.sum{mobile_ct[next_node_name] + T_mm[next_node_name]}
    for branch_info in branch_results:
        next_node_name = branch_info['next_node_name']
        T_mm = branch_info["T_mm"]
        T_mm_path = str(branch_info["T_mm_path"])
        if next_node_name == branch_end_node_name:
            mobile_begin_min_value += T_mm
        else:
            mobile_begin_min_value += nodes[next_node_name].mobile_ct + T_mm
        mobile_begin_min_path.append((next_node_name, 'mobile'))
        mobile_begin_min_path.append(T_mm_path)
    print(mobile_begin_min_value,mobile_begin_min_path)
    # b.upload_tt[branch_begin_node_name] + sum{ edge_ct[next_node_name] + T_em[next_node_name]}
    value = 0
    path = []
    value += nodes[branch_begin_node_name].upload_tt
    for branch_info in branch_results:
        next_node_name = branch_info['next_node_name']
        T_em = branch_info["T_em"]
        T_em_path = str(branch_info["T_em_path"])
        if next_node_name == branch_end_node_name:
            value += T_em
        else:
            value += nodes[next_node_name].edge_ct + T_em
        path.append((next_node_name, 'edge'))
        path.append(T_em_path)
    print(value,path)
    if value < mobile_begin_min_value:
        mobile_begin_min_value = value
        mobile_begin_min_path = path
    # c.upload_tt[branch_begin_node_name] + sum{ min( mobile_ct[next_node_name] + T_mm[next_node_name] ,edge_ct[next_node_name] + T_em[next_node_name])}
    value = 0
    path = []
    value += nodes[branch_begin_node_name].upload_tt
    for branch_info in branch_results:
        next_node_name = branch_info['next_node_name']
        T_mm = branch_info["T_mm"]
        T_mm_path = str(branch_info["T_mm_path"])
        T_em = branch_info["T_em"]
        T_em_path = str(branch_info["T_em_path"])
        if next_node_name == branch_end_node_name:
            if  T_mm < T_em:
                value += T_mm
                path.append((next_node_name, 'mobile'))
                path.append(T_mm_path)
            else:
                value += T_em
                path.append((next_node_name, 'edge'))
                path.append(T_em_path)
        else:
            if nodes[next_node_name].mobile_ct + T_mm < nodes[next_node_name].edge_ct + T_em:
                value += nodes[next_node_name].mobile_ct + T_mm
                path.append((next_node_name, 'mobile'))
                path.append(T_mm_path)
            else:
                value += nodes[next_node_name].edge_ct + T_em
                path.append((next_node_name, 'edge'))
                path.append(T_em_path)
    print(value, path)
    if value < mobile_begin_min_value:
        mobile_begin_min_value = value
        mobile_begin_min_path = path
    # 1.2 分支开始节点在边缘设备
    edge_begin_min_value = 0
    edge_begin_min_path = []
    # a.sum{edge_ct[next_node_name] + T_em[next_node_name]}
    for branch_info in branch_results:
        next_node_name = branch_info['next_node_name']
        T_em = branch_info["T_em"]
        T_em_path = str(branch_info["T_em_path"])
        if next_node_name == branch_end_node_name:
            edge_begin_min_value += T_em
        else:
            edge_begin_min_value += nodes[next_node_name].edge_ct + T_em
        edge_begin_min_path.append((next_node_name, 'edge'))
        edge_begin_min_path.append(T_em_path)
    print(edge_begin_min_value,edge_begin_min_path)
    # b.download_tt[branch_begin_node_name] + sum{mobile_ct[next_node_name] + T_mm[next_node_name]}
    value = 0
    path = []
    value += nodes[branch_begin_node_name].download_tt
    for branch_info in branch_results:
        next_node_name = branch_info['next_node_name']
        T_mm = branch_info["T_mm"]
        T_mm_path = str(branch_info["T_mm_path"])
        if next_node_name == branch_end_node_name:
            value += T_mm
        else:
            value += nodes[next_node_name].mobile_ct + T_mm
        path.append((next_node_name, 'mobile'))
        path.append(T_mm_path)
    print(value,path)
    if value < edge_begin_min_value:
        edge_begin_min_value = value
        edge_begin_min_path = path
    # c.download_tt[branch_begin_node_name] + sum{min(mobile_ct[next_node_name] + T_mm[next_node_name],edge_ct[next_node_name] + T_em[next_node_name])}
    value = 0
    path = []
    value += nodes[branch_begin_node_name].download_tt
    for branch_info in branch_results:
        next_node_name = branch_info['next_node_name']
        T_mm = branch_info["T_mm"]
        T_mm_path = str(branch_info["T_mm_path"])
        T_em = branch_info["T_em"]
        T_em_path = str(branch_info["T_em_path"])
        if next_node_name == branch_end_node_name:
            if T_mm < T_em:
                value += T_mm
                path.append((next_node_name, 'mobile'))
                path.append(T_mm_path)
            else:
                value += T_em
                path.append((next_node_name, 'edge'))
                path.append(T_em_path)
        else:
            if nodes[next_node_name].mobile_ct + T_mm < nodes[next_node_name].edge_ct + T_em:
                value += nodes[next_node_name].mobile_ct + T_mm
                path.append((next_node_name, 'mobile'))
                path.append(T_mm_path)
            else:
                value += nodes[next_node_name].edge_ct + T_em
                path.append((next_node_name, 'edge'))
                path.append(T_em_path)
    print(value, path)
    if value < edge_begin_min_value:
        edge_begin_min_value = value
        edge_begin_min_path = path

    dp_mobile[branch_end_node_name] = min(dp_mobile[branch_begin_node_name] + mobile_begin_min_value + nodes[branch_end_node_name].mobile_ct,
                                          dp_edge[branch_begin_node_name] + edge_begin_min_value + nodes[branch_end_node_name].mobile_ct)
    print("s_mobile: %.4f  = min(%.4f + %.4f + %.4f,%.4f + %.4f + %.4f)"
          % (dp_mobile[branch_end_node_name],
             dp_mobile[branch_begin_node_name], mobile_begin_min_value, nodes[branch_end_node_name].mobile_ct,
             dp_edge[branch_begin_node_name], edge_begin_min_value, nodes[branch_end_node_name].mobile_ct))
    if dp_mobile[branch_begin_node_name] + mobile_begin_min_value + nodes[branch_end_node_name].mobile_ct < dp_edge[branch_begin_node_name] + edge_begin_min_value + nodes[branch_end_node_name].mobile_ct:
        path_mobile[branch_end_node_name] = [(branch_begin_node_name,'mobile')] + mobile_begin_min_path
    else:
        path_mobile[branch_end_node_name] = [(branch_begin_node_name, 'edge')] + edge_begin_min_path

    ''' 2. 分支结束节点在边缘设备执行 '''
    ### 2.1 分支开始节点在边缘设备
    edge_begin_min_value = 0
    edge_begin_min_path = []
    # a. sum{ edge_ct[next_node_name] + T_ee[next_node_name]}
    for branch_info in branch_results:
        next_node_name = branch_info['next_node_name']
        T_ee = branch_info["T_ee"]
        T_ee_path = str(branch_info["T_ee_path"])
        if next_node_name == branch_end_node_name:
            edge_begin_min_value += T_ee
        else:
            edge_begin_min_value += nodes[next_node_name].edge_ct + T_ee
        edge_begin_min_path.append((next_node_name, 'edge'))
        edge_begin_min_path.append(T_ee_path)
    print(edge_begin_min_value, edge_begin_min_path)
    # b.download_tt[branch_begin_node_name] + sum{ mobile_ct[next_node_name] + T_me[next_node_name]}
    value = 0
    path = []
    value += nodes[branch_begin_node_name].download_tt
    for branch_info in branch_results:
        next_node_name = branch_info['next_node_name']
        T_me = branch_info["T_me"]
        T_me_path = str(branch_info["T_me_path"])
        if next_node_name == branch_end_node_name:
            value += T_me
        else:
            value += nodes[next_node_name].mobile_ct + T_me
        path.append((next_node_name, 'mobile'))
        path.append(T_me_path)
    print(value,path)
    if value < edge_begin_min_value:
        edge_begin_min_value = value
        edge_begin_min_path = path
    # c.download_tt[branch_begin_node_name] + sum{ min(edge_ct[next_node_name] + T_ee[next_node_name] ,mobile_ct[next_node_name] + T_me[next_node_name])}
    value = 0
    path = []
    value += nodes[branch_begin_node_name].download_tt
    for branch_info in branch_results:
        next_node_name = branch_info['next_node_name']
        T_me = branch_info["T_me"]
        T_me_path = str(branch_info["T_me_path"])
        T_ee = branch_info["T_ee"]
        T_ee_path = str(branch_info["T_ee_path"])
        if next_node_name == branch_end_node_name:
            if T_ee < T_me:
                value += T_ee
                path.append((next_node_name, 'edge'))
                path.append(T_ee_path)
            else:
                value += T_me
                path.append((next_node_name, 'mobile'))
                path.append(T_me_path)
        else:
            if nodes[next_node_name].edge_ct + T_ee < nodes[next_node_name].mobile_ct + T_me:
                value += nodes[next_node_name].edge_ct + T_ee
                path.append((next_node_name, 'edge'))
                path.append(T_ee_path)
            else:
                value += nodes[next_node_name].mobile_ct + T_me
                path.append((next_node_name, 'mobile'))
                path.append(T_me_path)
    print(value, path)
    if value < edge_begin_min_value:
        edge_begin_min_value = value
        edge_begin_min_path = path

    ### 2.2 分支开始节点在端设备
    mobile_begin_min_value = 0
    mobile_begin_min_path = []
    # a.sum{mobile_ct[next_node_name] + T_me[next_node_name]}
    for branch_info in branch_results:
        next_node_name = branch_info['next_node_name']
        T_me = branch_info["T_me"]
        T_me_path = str(branch_info["T_me_path"])
        if next_node_name == branch_end_node_name:
            mobile_begin_min_value += T_me
        else:
            mobile_begin_min_value += nodes[next_node_name].mobile_ct + T_me
        mobile_begin_min_path.append((next_node_name, 'mobile'))
        mobile_begin_min_path.append(T_me_path)
    print(mobile_begin_min_value,mobile_begin_min_path)
    # b.upload_tt[branch_begin_node_name] + sum{edge_ct[next_node_name] + T_ee[next_node_name]}
    value = 0
    path = []
    value += nodes[branch_begin_node_name].upload_tt
    for branch_info in branch_results:
        next_node_name = branch_info['next_node_name']
        T_ee = branch_info["T_ee"]
        T_ee_path = str(branch_info["T_ee_path"])
        if next_node_name == branch_end_node_name:
            value += T_ee
        else:
            value += nodes[next_node_name].edge_ct + T_ee
        path.append((next_node_name, 'edge'))
        path.append(T_ee_path)
    print(value, path)
    if value < mobile_begin_min_value:
        mobile_begin_min_value = value
        mobile_begin_min_path = path
    # c.upload_tt[branch_begin_node_name] + sum{ min(edge_ct[next_node_name] + T_ee[next_node_name] ,mobile_ct[next_node_name] + T_me[next_node_name])}
    value = 0
    path = []
    value += nodes[branch_begin_node_name].upload_tt
    for branch_info in branch_results:
        next_node_name = branch_info['next_node_name']
        T_me = branch_info["T_me"]
        T_me_path = str(branch_info["T_me_path"])
        T_ee = branch_info["T_ee"]
        T_ee_path = str(branch_info["T_ee_path"])
        if next_node_name == branch_end_node_name:
            if T_ee < T_me:
                value +=  T_ee
                path.append((next_node_name, 'edge'))
                path.append(T_ee_path)
            else:
                value += T_me
                path.append((next_node_name, 'mobile'))
                path.append(T_me_path)
        else:
            if nodes[next_node_name].edge_ct + T_ee < nodes[next_node_name].mobile_ct + T_me:
                value += nodes[next_node_name].edge_ct + T_ee
                path.append((next_node_name, 'edge'))
                path.append(T_ee_path)
            else:
                value += nodes[next_node_name].mobile_ct + T_me
                path.append((next_node_name, 'mobile'))
                path.append(T_me_path)
    print(value, path)
    if value < mobile_begin_min_value:
        mobile_begin_min_value = value
        mobile_begin_min_path = path

    dp_edge[branch_end_node_name] = min(
        dp_edge[branch_begin_node_name] + edge_begin_min_value + nodes[branch_end_node_name].edge_ct,
        dp_mobile[branch_begin_node_name] + mobile_begin_min_value + nodes[branch_end_node_name].edge_ct)
    print("s_edge: %.4f  = min(%.4f + %.4f + %.4f,%.4f + %.4f + %.4f)"
          % (dp_edge[branch_end_node_name],
             dp_edge[branch_begin_node_name], edge_begin_min_value, nodes[branch_end_node_name].edge_ct,
             dp_mobile[branch_begin_node_name],mobile_begin_min_value, nodes[branch_end_node_name].edge_ct))
    if dp_edge[branch_begin_node_name] + edge_begin_min_value + nodes[branch_end_node_name].edge_ct < \
            dp_mobile[branch_begin_node_name] + mobile_begin_min_value + nodes[branch_end_node_name].edge_ct:
        path_edge[branch_end_node_name] = [(branch_begin_node_name, 'edge')] + edge_begin_min_path
    else:
        path_edge[branch_end_node_name] = [(branch_begin_node_name, 'mobile')] + mobile_begin_min_path

    print('path_mobile[branch_end_node_name] ',path_mobile[branch_end_node_name])
    print('path_edge[branch_end_node_name] ',path_edge[branch_end_node_name])

    return branch_end_node_name

"""
非线性拓扑分支算法 => 输入的DNN只能有一个出口
"""
def nonlinear_partition_algorithm(nodes,S,T,input_in_mobile=True):
    '''
        input_in_mobile: 表示原始输入数据是否在mobile
        d_mobile: 状态结束时在移动设备执行的用时
        d_edge: 状态结束时在边缘设备执行的用时
    '''
    # for node in nodes:
    #     print(node)
    print()
    print('call for nonlinear_partition_algorithm,input_in_mobile = ',input_in_mobile)
    dp_mobile = {}
    dp_edge = {}
    path_mobile = {}
    path_edge = {}
    if input_in_mobile:
        dp_mobile[S.layer_name] = 0
        dp_edge[S.layer_name] = S.upload_tt
    else:
        dp_mobile[S.layer_name] = S.download_tt
        dp_edge[S.layer_name] = 0

    # 先假设S只有一个后继
    assert len(S.next_nodes) == 1
    current_node = nodes[S.next_nodes[0]]
    # while current_node.layer_name != "T":
    while current_node.layer_name != T.layer_name:
        current_node_name = current_node.layer_name

        dp_state_move_to_next(nodes, current_node, dp_mobile, path_mobile, dp_edge, path_edge)

        if len(current_node.next_nodes) == 1 :
            """ 线性拓扑节点 <= 当前节点只有一个后继 """
            print(current_node_name,'linear node , move to next node ...')
            # 下一个节点
            current_node = nodes[current_node.next_nodes[0]]

        else :
            """ 有多个后继、分支结构开始"""
            print(current_node_name, 'branch begin ...')
            # 二.移动到分支结束节点
            branch_end_node_name = dp_state_move_to_branch_end(nodes,
                                                               current_node, dp_mobile,
                                                               path_mobile, dp_edge, path_edge)
            branch_end_node = nodes[branch_end_node_name]
            # 下一个节点
            while branch_end_node.next_nodes and (len(branch_end_node.next_nodes) > 1):
                branch_end_node_name = dp_state_move_to_branch_end(nodes, branch_end_node,
                                                                   dp_mobile, path_mobile,
                                                                   dp_edge, path_edge)
                branch_end_node = nodes[branch_end_node_name]
            # 下一个节点 => 不是分支开始节点
            # if branch_end_node_name == "T":
            if branch_end_node_name == T.layer_name:
                print('!!!!!! branch_end_node_name == T.layer_name !!!!')
                # print('dp_edge[T.layer_name] ',dp_edge[T.layer_name],path_edge[T.layer_name],)
                # print('dp_mobile[T.layer_name] ',dp_mobile[T.layer_name],path_mobile[T.layer_name])
                return dp_edge[T.layer_name], path_edge, dp_mobile[T.layer_name], path_mobile
                break
            current_node = nodes[branch_end_node.next_nodes[0]]

    # current_node 现在应该是T节点
    if len(current_node.previous_nodes) == 1: # 只有一个最终输出
        last_node_name = current_node.previous_nodes[0]

        print('=' * 10, T.layer_name, '=' * 10)

        # 最终的结果提供给边缘
        T_end_in_edge = min(dp_edge[last_node_name], dp_mobile[last_node_name] + nodes[last_node_name].upload_tt)
        print("T_end_in_edge: %.4f  = min(%.4f ,%.4f + %.4f)"
              % (T_end_in_edge, dp_edge[last_node_name],
                 dp_mobile[last_node_name], nodes[last_node_name].upload_tt))
        if dp_edge[last_node_name] < dp_mobile[last_node_name] + nodes[last_node_name].upload_tt:
            path_edge[T.layer_name] = (last_node_name, 'edge')
        else:
            path_edge[T.layer_name] = (last_node_name,'mobile')

        # 最终的结果提供给mobile
        T_end_in_mobile = min(dp_mobile[last_node_name], dp_edge[last_node_name] + nodes[last_node_name].download_tt)
        print("T_end_in_mobile: %.4f  = min(%.4f ,%.4f + %.4f)"
              % (T_end_in_mobile, dp_mobile[last_node_name],
                 dp_edge[last_node_name], nodes[last_node_name].download_tt))
        if dp_mobile[last_node_name] < dp_edge[last_node_name] + nodes[last_node_name].download_tt:
            # path_mobile['T'] = (last_node_name,'mobile')
            path_mobile[T.layer_name] = (last_node_name,'mobile')
        else:
            # path_mobile['T'] = (last_node_name, 'edge')
            path_mobile[T.layer_name] = (last_node_name, 'edge')

        return T_end_in_edge, path_edge, T_end_in_mobile, path_mobile
    else:
        # return dp_edge["T"],path_edge["T"],dp_mobile["T"],path_mobile["T"]
        return dp_edge[T.layer_name],path_edge,dp_mobile[T.layer_name],path_mobile

def path_track_back(path_edge, path_mobile,nodes,end_mobile=True,end_name='T',print_info=True):
    track_info = ''
    if print_info:
        print('+' * 15,'path_track_back','+' * 15)
    if end_mobile:
        last_node_name = end_name
        run_place = 'mobile'
    else:
        last_node_name = end_name
        run_place = 'edge'
    while nodes[last_node_name].previous_nodes != None:
        if run_place == 'edge':
            used_dic = path_edge
        else:
            used_dic = path_mobile
        if len(nodes[last_node_name].previous_nodes) == 1:
            run_place = used_dic[last_node_name][1]
            last_node_name = used_dic[last_node_name][0]
            if print_info:
                print(last_node_name, run_place)
            track_info += last_node_name + ' ' + run_place  + '\n'
        else:
            # 此时的last_node_name是分支结束节点 => 跳到分支开始节点
            # {分支结束节点:((分支开始节点:place),(分支1起始节点:place),"{分支1}",(分支2起始节点:place),"{分支2}"....)}
            if print_info:
                print('-' * 10,'branch end node',last_node_name,'-' * 10)
            track_info += ('-' * 10) + 'branch end node'+ last_node_name + ('-' * 10) +'\n'
            # print('last_node_name ',last_node_name,
            #       ' run_place ',run_place,used_dic[last_node_name])
            i = 1
            while i < len(used_dic[last_node_name]):
                # print('first_succeed_node',used_dic[last_node_name][i],'after_this_node',used_dic[last_node_name][i+1])
                # track_info += 'first_succeed_node' + str(used_dic[last_node_name][i]) + 'after_this_node'+str(used_dic[last_node_name][i+1])+','
                if print_info:
                    print(used_dic[last_node_name][i+1])
                track_info += str(used_dic[last_node_name][i+1])
                i += 2
            run_place = used_dic[last_node_name][0][1]
            last_node_name = used_dic[last_node_name][0][0]
            if print_info:
                print('-' * 10, 'branch_begin_node', last_node_name, run_place, '-' * 10)
            track_info += ('-' * 10)+'branch_begin_node'+ last_node_name+ run_place+('-' * 10)+'\n'

    return track_info

if __name__ == '__main__':
    #VGG
    nodes, S = construct_DNN_topology(Model_Path="models/VGG/", main_target_name='predictions',mode='3G')

    """分区算法执行"""
    print('+'*100)
    print('my_partition Algo run...')
    T_end_in_edge, path_edge, T_end_in_mobile, path_mobile = nonlinear_partition_algorithm(nodes, S,nodes['T'])
    print()
    print('result track back.....')
    path_track_back(path_edge, path_mobile,nodes)

    print('+' * 100)
    print('final result')
    print('T_end_in_edge', T_end_in_edge, path_edge)
    print('T_end_in_mobile', T_end_in_mobile, path_mobile)