# coding=utf-8
from functools import reduce
import copy
from keras.models import load_model
from Utils import load_tt_dict

class Node():
    def __init__(self,layer_name,next_nodes,previous_nodes,
                 edge_ct,mobile_ct,upload_tt,download_tt,out_size,
                 out_shape,index):

        self.layer_name = layer_name

        self.next_nodes = next_nodes

        self.previous_nodes = previous_nodes

        self.edge_ct = edge_ct

        self.mobile_ct = mobile_ct

        self.upload_tt = upload_tt

        self.download_tt = download_tt

        self.out_size = out_size

        self.out_shape = out_shape

        self.index = index

    def __str__(self):
        obj_str = ""
        obj_str += "layer_name: "+self.layer_name+'\t'
        obj_str += "self.next_nodes: "+str(self.next_nodes)+'\t'
        obj_str += "previous_nodes: "+str(self.previous_nodes)+'\t'
        obj_str += "edge_ct: "+str(self.edge_ct)+'\t'
        obj_str += "mobile_ct: "+str(self.mobile_ct)+'\t'
        obj_str += "upload_tt: "+str(self.upload_tt)+'\t'
        obj_str += "download_tt: "+str(self.download_tt)+'\t'
        obj_str += "out_size: "+str(self.out_size)+'\t'
        obj_str += "out_shape: "+str(self.out_shape)
        return obj_str

class MyDNNNet():
    def __init__(self,edge_ct_path,mobile_ct_path,
                 upload_tt_path,download_tt_path,
                 h5_model_path='model.h5',main_target_name=None):
        self.h5_model_path = h5_model_path
        self.edge_ct_path = edge_ct_path
        self.mobile_ct_path = mobile_ct_path
        self.upload_tt_path = upload_tt_path
        self.download_tt_path = download_tt_path
        self.S = None
        self.T = None
        self.nodes = {}
        self.main_target_name = main_target_name


    def transform_topology(self,mode):

        bandwidth_dict = {'3G':[1.1,4.6],
                          '4G':[8.8,29.1],
                          'WIFI':[18.88,54.97],
                          'execute':[None,None]
                          }

        dnn_net = load_model(self.h5_model_path)

        edge_ct_list = [1 * float(x.strip().split('\t')[1]) for x in open(self.edge_ct_path,'r',encoding='utf-8').readlines()]
        print('edge_ct_list len ',len(edge_ct_list),edge_ct_list[:5])

        mobile_ct_list = [ 1 * float(x.strip().split('\t')[1]) for x in open(self.mobile_ct_path,'r',encoding='utf-8').readlines()]
        print('mobile_ct_list len ',len(mobile_ct_list),mobile_ct_list[:5])

        upload_tt_dic = load_tt_dict(self.upload_tt_path,bandwidth_dict[mode][0])
        print('upload_tt_dic ',upload_tt_dic)

        download_tt_dic = load_tt_dict(self.download_tt_path,bandwidth_dict[mode][1])
        print('download_tt_dic ',download_tt_dic)


        #layer_name,next_nodes,previous_nodes,
        # edge_ct,mobile_ct,upload_tt,download_tt
        index = 0
        for layer in dnn_net.layers:
            out_shape = [1] + list(layer.output_shape[1:])
            out_size = int(reduce(lambda x, y: x * y, layer.output_shape[1:], 1))

            if layer.name == 'input_1':

                self.S = Node(layer_name=layer.name,
                              next_nodes=[],previous_nodes=None,
                              edge_ct=0,mobile_ct=0,
                              upload_tt=upload_tt_dic[out_size],download_tt=download_tt_dic[out_size],
                              out_size=out_size,out_shape=out_shape,index=index)
                self.nodes[layer.name] = self.S
            else:

                previous_nodes = []

                for node in layer._inbound_nodes:
                    for i in range(len(node.inbound_layers)):
                        inbound_layer = node.inbound_layers[i].name
                        previous_nodes.append(inbound_layer)

                simple_node = Node(layer_name=layer.name,
                                   next_nodes=[],previous_nodes=previous_nodes,
                                   edge_ct=edge_ct_list[index],mobile_ct=mobile_ct_list[index],
                                   upload_tt=upload_tt_dic[out_size],download_tt=download_tt_dic[out_size],
                                   out_size=out_size, out_shape=out_shape,index=index)


                for pre_node_name in previous_nodes:
                    self.nodes[pre_node_name].next_nodes.append(layer.name)

                self.nodes[layer.name] = simple_node

            index += 1

        T_previous_nodes = []
        for name,node in self.nodes.items():
            if len(node.next_nodes) == 0:
                T_previous_nodes.append(name)
        for T_pre_node in T_previous_nodes:
            self.nodes[T_pre_node].next_nodes.append("T")


        self.T = Node(layer_name="T",
                      next_nodes=None, previous_nodes=T_previous_nodes,
                      edge_ct=0, mobile_ct=0,
                      upload_tt=0, download_tt=0,
                      out_size=0, out_shape=None,index=index)

        self.nodes[self.T.layer_name] = self.T

    def transform_rnn_topology(self,mode,sequence_len = 256):

        bandwidth_dict = {'3G':[1.1,4.6],
                          '4G':[8.8,29.1],
                          'WIFI':[18.88,54.97],
                          'execute':[None,None]
                          }

        dnn_net = load_model(self.h5_model_path)

        edge_ct_list = [1 * float(x.strip().split('\t')[1]) for x in open(self.edge_ct_path,'r',encoding='utf-8').readlines()]
        print('edge_ct_list len ',len(edge_ct_list),edge_ct_list[:5])

        mobile_ct_list = [ 1 * float(x.strip().split('\t')[1]) for x in open(self.mobile_ct_path,'r',encoding='utf-8').readlines()]
        print('mobile_ct_list len ',len(mobile_ct_list),mobile_ct_list[:5])

        upload_tt_dic = load_tt_dict(self.upload_tt_path,bandwidth_dict[mode][0])
        print('upload_tt_dic ',upload_tt_dic)

        download_tt_dic = load_tt_dict(self.download_tt_path,bandwidth_dict[mode][1])
        print('download_tt_dic ',download_tt_dic)


        #layer_name,next_nodes,previous_nodes,
        # edge_ct,mobile_ct,upload_tt,download_tt
        index = 0
        for layer in dnn_net.layers:

            out_shape = [1, sequence_len] + list(layer.output_shape[2:])
            out_size = reduce(lambda x, y: x * y, layer.output_shape[2:], sequence_len)

            if layer.name == 'input_1':

                self.S = Node(layer_name=layer.name,
                              next_nodes=[],previous_nodes=None,
                              edge_ct=0,mobile_ct=0,
                              upload_tt=upload_tt_dic[out_size],download_tt=download_tt_dic[out_size],
                              out_size=out_size,out_shape=out_shape,index=index)
                self.nodes[layer.name] = self.S
            else:

                previous_nodes = []
                for node in layer._inbound_nodes:
                    for i in range(len(node.inbound_layers)):
                        inbound_layer = node.inbound_layers[i].name
                        previous_nodes.append(inbound_layer)

                simple_node = Node(layer_name=layer.name,
                                   next_nodes=[],previous_nodes=previous_nodes,
                                   edge_ct=edge_ct_list[index],mobile_ct=mobile_ct_list[index],
                                   upload_tt=upload_tt_dic[out_size],download_tt=download_tt_dic[out_size],
                                   out_size=out_size, out_shape=out_shape,index=index)


                for pre_node_name in previous_nodes:
                    self.nodes[pre_node_name].next_nodes.append(layer.name)

                self.nodes[layer.name] = simple_node

            index += 1

        T_previous_nodes = []
        for name,node in self.nodes.items():
            if len(node.next_nodes) == 0:
                T_previous_nodes.append(name)
        for T_pre_node in T_previous_nodes:
            self.nodes[T_pre_node].next_nodes.append("T")


        self.T = Node(layer_name="T",
                      next_nodes=None, previous_nodes=T_previous_nodes,
                      edge_ct=0, mobile_ct=0,
                      upload_tt=0, download_tt=0,
                      out_size=0, out_shape=None,index=index)

        self.nodes[self.T.layer_name] = self.T

    def is_multi_output(self):
        return len(self.T.previous_nodes) > 1

    def handle_multi_output(self):

        self.multi_net = {}


        main_nodes_name = [self.main_target_name]
        queue = []
        queue.append(self.main_target_name)
        while len(queue) > 0:
            cur_name = queue[0]
            queue.remove(cur_name)
            if self.nodes[cur_name].previous_nodes != None:
                for pre_name in self.nodes[cur_name].previous_nodes:
                    if pre_name not in main_nodes_name:
                        queue.append(pre_name)
                        main_nodes_name.append(pre_name)
        print('len main_nodes_name',len(main_nodes_name))
        print('main node ',main_nodes_name[::-1])

        print('build main top')
        main_nodes = {}
        for name in main_nodes_name:
            # print(name)
            node = copy.deepcopy(self.nodes[name])
            new_next_nodes = []
            for next_name in node.next_nodes:
                if next_name in main_nodes_name:
                    new_next_nodes.append(next_name)
            node.next_nodes = new_next_nodes
            main_nodes[name] = node
        main_nodes[self.main_target_name].next_nodes = ["T"]
        main_nodes['T'] = Node(layer_name="T",
                      next_nodes=None, previous_nodes=[self.main_target_name],
                      edge_ct=0, mobile_ct=0,
                      upload_tt=0, download_tt=0,
                      out_size=0, out_shape=None,index=self.T.index)

        self.multi_net[self.main_target_name] = (main_nodes,self.S)

        auxiliary_targets_name = copy.deepcopy(self.T.previous_nodes)
        auxiliary_targets_name.remove(self.main_target_name)
        print('auxiliary_targets_name ',auxiliary_targets_name)
        for aux_target_name in auxiliary_targets_name:
            print('aux_target_name',aux_target_name)

            aux_nodes_name = [aux_target_name]
            queue = []
            queue.append(aux_target_name)
            while len(queue) > 0:
                cur_name = queue[0]
                queue.remove(cur_name)
                for pre_node in self.nodes[cur_name].previous_nodes:
                    if pre_node not in main_nodes_name and pre_node not in aux_nodes_name:
                        queue.append(pre_node)
                        aux_nodes_name.append(pre_node)
            print('len aux_nodes_name ',len(aux_nodes_name))
            print('aux_nodes_name ', aux_nodes_name[::-1])

            aux_nodes = {}
            for name in aux_nodes_name:
                # print(name)
                node = copy.deepcopy(self.nodes[name])
                new_previous_nodes = []
                for pre_name in node.previous_nodes:
                    if pre_name not in main_nodes_name:
                        new_previous_nodes.append(pre_name)
                node.previous_nodes = new_previous_nodes
                aux_nodes[name] = node
            aux_nodes[aux_target_name].next_nodes = ["T"]
            aux_nodes['T'] = Node(layer_name="T",
                                   next_nodes=None, previous_nodes=[aux_target_name],
                                   edge_ct=0, mobile_ct=0,
                                   upload_tt=0, download_tt=0,
                                   out_size=0, out_shape=None,index=self.T.index)
            self.multi_net[aux_target_name] = (aux_nodes, aux_nodes[aux_nodes_name[-1]])


def construct_DNN_topology(Model_Path ,main_target_name,mode,is_rnn=False):

    myDNNNet = MyDNNNet(edge_ct_path=Model_Path + 'EdgeNodeComputeTime.txt',
                        mobile_ct_path=Model_Path + 'MobileNodeComputeTime.txt',
                        upload_tt_path=Model_Path + 'MobileNodeUploadTime.txt',
                        download_tt_path=Model_Path + 'MobileNodeDownloadTime.txt',
                        h5_model_path=Model_Path + 'model.h5', main_target_name=main_target_name)
    if not is_rnn:
        myDNNNet.transform_topology(mode)
    else:
        myDNNNet.transform_rnn_topology(mode)

    # for name, node in myDNNNet.nodes.items():
    #     print(node)

    if myDNNNet.is_multi_output():

        print(Model_Path,'model has multi outputs...')
        myDNNNet.handle_multi_output()
        main_nodes, main_S = myDNNNet.multi_net[main_target_name]
    else:
        print(Model_Path, 'model has only one output...')
        main_nodes, main_S = myDNNNet.nodes,myDNNNet.S

    for name, node in main_nodes.items():
        print(node)

    sum_edge_time = 0
    sum_mobile_time = 0
    for name, node in main_nodes.items():
        sum_edge_time += node.edge_ct
        sum_mobile_time += node.mobile_ct
    print('sum_edge_time', sum_edge_time, 'sum_mobile_time', sum_mobile_time)
    print('upload input ', main_S.upload_tt)

    return main_nodes, main_S

"""
Chairs
"""
def construct_multi_input_DNN_topology(Model_Path ,main_target_name,mode):

    myDNNNet = MyDNNNet(edge_ct_path=Model_Path + 'EdgeNodeComputeTime.txt',
                        mobile_ct_path=Model_Path + 'MobileNodeComputeTime.txt',
                        upload_tt_path=Model_Path + 'MobileNodeUploadTime.txt',
                        download_tt_path=Model_Path + 'MobileNodeDownloadTime.txt',
                        h5_model_path=Model_Path + 'model.h5', main_target_name=main_target_name)
    myDNNNet.transform_topology(mode)
    # for name, node in myDNNNet.nodes.items():
    #     print(node)


    if myDNNNet.is_multi_output():

        print(Model_Path,'model has multi outputs...')
        myDNNNet.handle_multi_output()
        main_nodes, main_S = myDNNNet.multi_net[main_target_name]
    else:
        print(Model_Path, 'model has only one output...')
        main_nodes, main_S = myDNNNet.nodes,myDNNNet.S


    input_nodes = []
    input_upload_tt = 0
    input_download_tt = 0
    for name, node in main_nodes.items():
        if len(node.previous_nodes) == 0:
            print('input name ',name)
            input_nodes.append(name)
            input_upload_tt += node.upload_tt
            input_download_tt += node.download_tt
            node.previous_nodes = ['input_next_node']

    S_next_node = Node(layer_name='input_next_node',
                       next_nodes=input_nodes, previous_nodes=['input_1'],
                       edge_ct=0, mobile_ct=0,
                       upload_tt=0, download_tt=0,
                       out_size=0, out_shape=[], index=-1)
    main_S = Node(layer_name='input_1',
                  next_nodes=[S_next_node.layer_name], previous_nodes=None,
                  edge_ct=0, mobile_ct=0,
                  upload_tt=input_upload_tt, download_tt=input_download_tt,
                  out_size=0, out_shape=[], index=-2)

    main_nodes['input_1'] = main_S
    main_nodes['input_next_node'] = S_next_node
    myDNNNet.S = main_S

    for name, node in main_nodes.items():
        print(node)

    sum_edge_time = 0
    sum_mobile_time = 0
    for name, node in main_nodes.items():
        sum_edge_time += node.edge_ct
        sum_mobile_time += node.mobile_ct
    print('sum_edge_time', sum_edge_time, 'sum_mobile_time', sum_mobile_time)
    print('upload input ', main_S.upload_tt)

    return main_nodes, main_S

