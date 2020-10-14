# coding=utf-8
from keras.models import load_model

def bandwidth_to_speed(bandwidth):
    bandwidth = bandwidth * 1.0# bandwidth => Mbps
    network_speed = bandwidth * 1024 / 8  # KB/s
    network_speed = network_speed * 1024  # B/s
    return network_speed

def load_tt_dict(path,bandwidth=None):

    if bandwidth:
        network_speed = bandwidth_to_speed(bandwidth)

    lines = open(path,'r',encoding='utf-8').readlines()
    lines = [x.strip() for x in lines]
    tt_dict = {}
    for line in lines:
        outsize,tt = line.split('\t')
        if bandwidth:
            tt_dict[int(outsize)] = float('%.4f' % (int(outsize) * 4 / network_speed * 1000)) # ms
        else:
            tt_dict[int(outsize)] = float(tt)

    return tt_dict

def get_model_dependence():

    layer_dependences = []
    TargetNet = load_model('F:\Graduation project\MasterCode\models\VGG\model.h5')

    for layer in TargetNet.layers:
        previous_nodes = []

        for node in layer._inbound_nodes:
            for i in range(len(node.inbound_layers)):
                inbound_layer = node.inbound_layers[i].name
                inbound_node_index = node.node_indices[i]
                inbound_tensor_index = node.tensor_indices[i]
                previous_nodes.append((inbound_layer,inbound_node_index,inbound_tensor_index))

        layer_dependences.append(previous_nodes)


    for i in range(len(layer_dependences)):
        print(i,layer_dependences[i])

    return layer_dependences

if __name__ == '__main__':
    out_size = 1024
    network_speed = bandwidth_to_speed(4.6)
    G3_u = float('%.4f' % (int(out_size) * 4 / network_speed * 1000))
    network_speed = bandwidth_to_speed(1.1)
    G3_d = float('%.4f' % (int(out_size) * 4 / network_speed * 1000))
    print(G3_u, G3_d)

    network_speed = bandwidth_to_speed(29.1)
    G4_u = float('%.4f' % (int(out_size) * 4 / network_speed * 1000))
    network_speed = bandwidth_to_speed(8.8)
    G4_d = float('%.4f' % (int(out_size) * 4 / network_speed * 1000))
    print(G4_u, G4_d)

    network_speed = bandwidth_to_speed(54.97)
    wifi_u = float('%.4f' % (int(out_size) * 4 / network_speed * 1000))
    network_speed = bandwidth_to_speed(18.88)
    wifi_d = float('%.4f' % (int(out_size) * 4 / network_speed * 1000))
    print(wifi_u, wifi_d)




