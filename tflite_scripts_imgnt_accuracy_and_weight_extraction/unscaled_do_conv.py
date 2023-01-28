from ctypes import util
import numpy as np
from models_archs import utils

utils.NET_PREFIX = 'mob_v2'
utils.set_globals(utils.NET_PREFIX, utils.NET_PREFIX)

layers_types = utils.read_layers_types()
layers_weights_shape = utils.read_layers_weight_shapes(layers_types)
layers_ifms_shape = utils.read_layers_input_shapes()
layers_strides = utils.read_layers_strides()
layers_relus = utils.read_layers_relus()

layers_ofms_shape = utils.read_layers_output_shapes()
skip_connections_indices = utils.read_skip_connections_indices()


layer_index = 0

layer_type = layers_types[layer_index]
weights_file = './{}/weights/conv2d_{}_{}_weights.txt'.format(
    utils.NET_PREFIX, layer_index, layer_type)
ifms_file = './{}/fms/fms_conv2d_{}_{}_{}_{}.txt'.format(utils.NET_PREFIX, layer_index, layers_ifms_shape[layer_index].depth, layers_ifms_shape[layer_index].height,
                                                layers_ifms_shape[layer_index].width)
#ifms_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/eff_b0/fms/fms_conv2d_1_mul_1_2_32_112_112.txt'
ofms_file = './scratch_out/ofms_{}_un.txt'.format(layer_index)


layers_ifms_zero_point = {layer_index: -127}

conv_strides = layers_strides[layer_index]

weights = np.loadtxt(weights_file).astype(np.int8)

print(weights[0:10])
if layer_type == 'dw':
    weights = np.reshape(weights, (layers_weights_shape[layer_index].num_of_filters, layers_weights_shape[layer_index].depth, layers_weights_shape[layer_index].height,
                                   layers_weights_shape[layer_index].width))
else:
    weights = np.reshape(weights, (layers_weights_shape[layer_index].num_of_filters, layers_weights_shape[layer_index].depth, layers_weights_shape[layer_index].height,
                                   layers_weights_shape[layer_index].width))

ifms = np.loadtxt(ifms_file).astype(np.int8)
ifms = np.reshape(ifms, (layers_ifms_shape[layer_index].depth,
                  layers_ifms_shape[layer_index].height, layers_ifms_shape[layer_index].width))

ofms_shape = (layers_weights_shape[layer_index].num_of_filters, int(layers_ifms_shape[layer_index].height /
                                                                    conv_strides), int(layers_ifms_shape[layer_index].width/conv_strides))
ofms = np.zeros(ofms_shape).astype(np.int32)

filter_height = layers_weights_shape[layer_index].height
filter_width = layers_weights_shape[layer_index].width
padding_val = int((filter_height - 1) / 2)
# print(layers_ifms_zero_point[layer_index])
if layer_type != 'pw':
    if conv_strides == 1:
        ifms = np.pad(ifms, ((0, 0), (padding_val, padding_val), (padding_val, padding_val)),
                      mode='constant', constant_values=layers_ifms_zero_point[layer_index])
    elif conv_strides == 2:
        ifms = np.pad(ifms, ((0, 0), (0, padding_val), (0, padding_val)),
                      mode='constant',  constant_values=layers_ifms_zero_point[layer_index])


def conv():
    for i in range(layers_weights_shape[layer_index].num_of_filters):
        for j in range(ofms_shape[1]):
            for k in range(ofms_shape[2]):
                tmp = np.sum(weights[i].astype(np.int32) * (ifms[:, j*conv_strides:j*conv_strides + filter_height,
                                                                 k*conv_strides:k*conv_strides + filter_width]))

                ofms[i][j][k] = tmp

                if i == 8 and j == 111 and k == 0:
                    print(ifms[:, j*conv_strides:j*conv_strides +
                          3, k*conv_strides:k*conv_strides+3])


def dw_conv():
    for i in range(layers_weights_shape[layer_index].num_of_filters):
        for j in range(ofms_shape[1]):
            for k in range(ofms_shape[2]):
                if i == 0 and j == 0 and k==0:
                    print(weights[i].astype(np.float32), ifms[i, j*conv_strides:j*conv_strides +
                                                               filter_height, k*conv_strides:k*conv_strides + filter_width])
                tmp = np.sum(weights[i].astype(np.float32) * ifms[i, j*conv_strides:j*conv_strides +
                                                                  filter_height, k*conv_strides:k*conv_strides + filter_width])

                ofms[i][j][k] = tmp


if layer_type == 'dw':
    dw_conv()
    print('dw')
else:
    conv()
ofms = ofms.reshape((ofms_shape[0] * ofms_shape[1] * ofms_shape[2]))
np.savetxt(ofms_file, ofms, fmt='%i')
