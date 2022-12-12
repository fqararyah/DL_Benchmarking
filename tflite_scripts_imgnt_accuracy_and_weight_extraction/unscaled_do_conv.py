from ctypes import util
import numpy as np
from models_archs import utils


layers_types = utils.read_layers_types()
layers_weights_shape = utils.read_layers_weight_shapes(layers_types)
layers_ifms_shape = utils.read_layers_input_shapes()
layers_strides = utils.read_layers_strides()
layers_relus = utils.read_layers_relus()

layers_ofms_shape = utils.read_layers_output_shapes()
skip_connections_indices = utils.read_skip_connections_indices()

tf_lite_to_my_cnn_layer_mapping = {0:1}
skip_connections_so_far = 0
for layer_index in range(1, len(layers_ofms_shape)):
    if layer_index in skip_connections_indices:
        skip_connections_so_far += 1
    tf_lite_to_my_cnn_layer_mapping[layer_index] = layer_index + skip_connections_so_far

layer_index = 4

layer_type = layers_types[layer_index]
weights_file = './weights/weights_{}_{}.txt'.format(layer_index, layer_type)
ifms_file = './fms/fms_{}_{}_{}_{}.txt'.format(tf_lite_to_my_cnn_layer_mapping[layer_index] if layer_index > 0 else 1, layers_ifms_shape[layer_index].depth, layers_ifms_shape[layer_index].height,\
    layers_ifms_shape[layer_index].width)
ofms_file = './scratch_out/ofms_{}_un.txt'.format(layer_index)

ifms_zero_points_file = './fms/fms_{}_zero_points.txt'.format(tf_lite_to_my_cnn_layer_mapping[layer_index] if layer_index > 0 else 1)

layers_ifms_zero_point = {layer_index: np.loadtxt(ifms_zero_points_file).astype(np.int32)}

conv_strides = layers_strides[layer_index]

weights = np.loadtxt(weights_file).astype(np.int8)
if layer_type == 'dw':
    weights = np.reshape(weights,(layers_weights_shape[layer_index].num_of_filters, layers_weights_shape[layer_index].depth, layers_weights_shape[layer_index].height,\
    layers_weights_shape[layer_index].width))
else:
    weights = np.reshape(weights,(layers_weights_shape[layer_index].num_of_filters, layers_weights_shape[layer_index].depth, layers_weights_shape[layer_index].height,\
    layers_weights_shape[layer_index].width))

ifms = np.loadtxt(ifms_file).astype(np.int8)
ifms = np.reshape(ifms, (layers_ifms_shape[layer_index].depth, layers_ifms_shape[layer_index].height, layers_ifms_shape[layer_index].width) )

ofms_shape = (layers_weights_shape[layer_index].num_of_filters, int(layers_ifms_shape[layer_index].height /
                                    conv_strides), int(layers_ifms_shape[layer_index].width/conv_strides))
ofms = np.zeros(ofms_shape).astype(np.int32)

filter_height =layers_weights_shape[layer_index].height
filter_width =layers_weights_shape[layer_index].width
padding_val = int( (filter_height - 1) / 2)
#print(layers_ifms_zero_point[layer_index])
if layer_type != 'pw':
    if conv_strides == 1:
        ifms = np.pad(ifms, ((0,0),(padding_val,padding_val),(padding_val,padding_val)), mode='constant', constant_values = layers_ifms_zero_point[layer_index])
    elif conv_strides == 2:
        ifms = np.pad(ifms, ((0,0),(0,padding_val),(0,padding_val)), mode='constant',  constant_values = layers_ifms_zero_point[layer_index])

def conv():
        for i in range(layers_weights_shape[layer_index].num_of_filters):
            for j in range(ofms_shape[1]):
                for k in range(ofms_shape[2]):
                    tmp = np.sum(weights[i].astype(np.int32) * ( ifms[:, j*conv_strides:j*conv_strides + filter_height, \
                        k*conv_strides:k*conv_strides + filter_width]) )
                    
                    ofms[i][j][k] = tmp

                    if i == 8 and j == 111 and k == 0:
                        print(ifms[:, j*conv_strides:j*conv_strides+3, k*conv_strides:k*conv_strides+3]) 

def dw_conv():
    for i in range(layers_weights_shape[layer_index].num_of_filters):
        for j in range(ofms_shape[1]):
            for k in range(ofms_shape[2]):
                tmp = np.sum(weights[i].astype(np.float32) * ifms[i, j*conv_strides:j*conv_strides + \
                        filter_height, k*conv_strides:k*conv_strides + filter_width])            
                
                ofms[i][j][k] = tmp

if layer_type == 'dw':
    dw_conv()
    print('dw')
else:
    conv()
ofms = ofms.reshape((ofms_shape[0]* ofms_shape[1]* ofms_shape[2]))
np.savetxt(ofms_file, ofms, fmt='%i')