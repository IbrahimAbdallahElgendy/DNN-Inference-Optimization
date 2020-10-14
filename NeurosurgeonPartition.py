# coding=utf-8
from DNNTopologyHandler import construct_DNN_topology

def Neuro_partition(nodes,S,T):

    accumulate_edge_ct = 0

    assert len(S.next_nodes) == 1
    current_node = nodes[S.next_nodes[0]]
    while current_node.layer_name != "T":
        assert len(current_node.previous_nodes) == 1
        accumulate_edge_ct += current_node.edge_ct

        assert len(current_node.next_nodes) == 1
        current_node = nodes[current_node.next_nodes[0]]
    min_end2end_infer_time = accumulate_edge_ct + S.upload_tt + nodes[T.previous_nodes[0]].download_tt
    min_node_name = "input_1"
    print('init min_end2end_infer_time: %.4f = %.4f + %.4f + %.4f'%(
        min_end2end_infer_time,
        accumulate_edge_ct,S.upload_tt,
        nodes[T.previous_nodes[0]].download_tt))

    accumulate_mobile_ct = 0
    assert len(S.next_nodes) == 1
    current_node = nodes[S.next_nodes[0]]
    while current_node.layer_name != "T":
        accumulate_mobile_ct += current_node.mobile_ct
        accumulate_edge_ct -= current_node.edge_ct
        end2end_infer_time = accumulate_mobile_ct + current_node.upload_tt + accumulate_edge_ct + nodes[T.previous_nodes[0]].download_tt
        print(current_node.layer_name,end2end_infer_time)
        if(end2end_infer_time < min_end2end_infer_time):
            min_end2end_infer_time = end2end_infer_time
            min_node_name  = current_node.layer_name
            print('new min_node_name',min_node_name)
            print('new min_end2end_infer_time: %.4f = %.4f + %.4f + %.4f +  %.4f'%(
                min_end2end_infer_time,
                accumulate_mobile_ct,current_node.upload_tt,
                accumulate_edge_ct,nodes[T.previous_nodes[0]].download_tt
            ))


        current_node = nodes[current_node.next_nodes[0]]

    return min_node_name,min_end2end_infer_time

if __name__ == '__main__':

    nodes, S = construct_DNN_topology(Model_Path="models/VGG/", main_target_name='predictions', mode='WIFI')

    print('+' * 100)
    print('Neuro_partition Algo run...')
    min_node_name, min_end2end_infer_time = Neuro_partition(nodes, S, nodes['T'])
    print('+' * 100)
    print('min_node_name',min_node_name,'min_end2end_infer_time',min_end2end_infer_time)