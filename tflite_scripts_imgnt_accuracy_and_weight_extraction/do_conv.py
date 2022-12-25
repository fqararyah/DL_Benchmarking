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

layer_index = 9

layer_type = layers_types[layer_index]
weights_file = './{}/weights/conv2d_{}_{}_weights.txt'.format(utils.NET_PREFIX, layer_index, layer_type)
ifms_file = './{}/fms/fms_conv2d_{}_{}_{}_{}.txt'.format(utils.NET_PREFIX,layer_index, layers_ifms_shape[layer_index].depth, layers_ifms_shape[layer_index].height,\
    layers_ifms_shape[layer_index].width)
ofms_file = './scratch_out/ofms_{}_ref.txt'.format(layer_index)
ifms_zero_points_file = './{}/fms/fms_conv2d_{}_zero_points.txt'.format(utils.NET_PREFIX, layer_index)
bias_file = './{}/biases/conv2d_{}_biases.txt'.format(utils.NET_PREFIX, layer_index)
weights_scale_file = './{}/weights/conv2d_{}_scales.txt'.format(utils.NET_PREFIX, layer_index)
ifms_scale_file = './{}/fms/fms_conv2d_{}_scales.txt'.format(utils.NET_PREFIX, layer_index)
ofms_scale_file = './{}/fms/fms_conv2d_{}_scales.txt'.format(utils.NET_PREFIX, layer_index + 1)
ofms_zero_points_file = './{}/fms/fms_conv2d_{}_zero_points.txt'.format(utils.NET_PREFIX, layer_index + 1)

layers_ifms_zero_point = {layer_index: np.loadtxt(ifms_zero_points_file).astype(np.int32)}
layers_bias = {layer_index: np.loadtxt(bias_file).astype(np.int32)}
layers_scale_ifms = {layer_index: np.loadtxt(ifms_scale_file)}#{0: 0.003921568393707275, 3: 0.0235294122248888, 6: 0.0235294122248888, 4:0.3023846447467804, 5: 0.00235294122248888 }
layers_scale_weights = {layer_index: np.loadtxt(weights_scale_file)}#{0: [0.0095633129], 3: [0.02902807], 6: [0.01043679], 4: [0.00100364], 5: [0.00320544]}
layers_scale_ofms = {layer_index: np.loadtxt(ofms_scale_file)}#{0: 0.0235294122248888, 3: 0.3023846447467804,  6: 0.1985088586807251, 4:0.0235294122248888, 5: 0.0235294122248888 } 
layers_ofms_zero_point = {layer_index: np.loadtxt(ofms_zero_points_file).astype(np.int8)}#{0: 128, 3: 6, 6: 5, 4: 128}
#layers_ifms_zero_point = {0: 128, 3: 128, 6: 128, 4: 6, 5:128}#{layer_index: np.loadtxt(ifms_zero_points_file).astype(np.int8)}
#layers_bias = {0: [61864]*32, 3: [-2630]*32, 6: [32910]*32, 4: [2650]*32}#{layer_index: np.loadtxt(bias_file).astype(np.int32)}
print(layers_ifms_zero_point[layer_index])
print(layers_ofms_zero_point[layer_index])
print(layers_bias[layer_index])
print(layers_scale_ifms[layer_index])
print(layers_scale_weights[layer_index])
print(layers_scale_ofms[layer_index])
#layers_ofms_zero_point
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
ofms = np.zeros(ofms_shape).astype(np.int8)

filter_height =layers_weights_shape[layer_index].height
filter_width =layers_weights_shape[layer_index].width
padding_val = int( (filter_height - 1) / 2)
#print(layers_ifms_zero_point[layer_index])
print(layers_scale_ifms[layer_index] * layers_scale_weights[layer_index][0])
if layer_type != 'pw':
    if conv_strides == 1:
        ifms = np.pad(ifms, ((0,0),(padding_val,padding_val),(padding_val,padding_val)), mode='constant', constant_values = layers_ifms_zero_point[layer_index])
    elif conv_strides == 2:
        ifms = np.pad(ifms, ((0,0),(0,padding_val),(0,padding_val)), mode='constant',  constant_values = layers_ifms_zero_point[layer_index])

def conv():
        for i in range(layers_weights_shape[layer_index].num_of_filters):
            for j in range(ofms_shape[1]):
                for k in range(ofms_shape[2]):
                    tmp = np.sum(weights[i].astype(np.int32)) * -layers_ifms_zero_point[layer_index] + \
                    np.sum(weights[i].astype(np.int32) * ( ifms[:, j*conv_strides:j*conv_strides + filter_height, \
                        k*conv_strides:k*conv_strides + filter_width]) ) + layers_bias[layer_index][i]
                    
                    tmp = tmp * layers_scale_ifms[layer_index] * layers_scale_weights[layer_index][i]
                    if layers_relus[layer_index] == 6:
                        tmp = min(max(tmp, 0), 6)
                    tmp = tmp / layers_scale_ofms[layer_index] + layers_ofms_zero_point[layer_index]
                    if tmp > 0:
                        tmp = int(tmp + 0.5)
                    else:
                        tmp = int(tmp -0.5)
                    ofms[i][j][k] = tmp

def dw_conv():
    for i in range(layers_weights_shape[layer_index].num_of_filters):
        for j in range(ofms_shape[1]):
            for k in range(ofms_shape[2]):
                tmp = np.sum(weights[i].astype(np.float32)) * - layers_ifms_zero_point[layer_index] + \
                    np.sum(weights[i].astype(np.float32) * ifms[i, j*conv_strides:j*conv_strides + \
                        filter_height, k*conv_strides:k*conv_strides + filter_width]) + layers_bias[layer_index][i]

                tmp = tmp * layers_scale_ifms[layer_index] * layers_scale_weights[layer_index][i]
                if layers_relus[layer_index] == 6:
                    tmp = min(max(tmp, 0), 6)
                tmp = tmp / layers_scale_ofms[layer_index] + layers_ofms_zero_point[layer_index]
                if tmp > 0:
                    tmp = int(tmp + 0.5)
                else:
                    tmp = int(tmp -0.5)
                ofms[i][j][k] = tmp

if layer_type == 'dw':
    dw_conv()
else:
    conv()
ofms = ofms.reshape((ofms_shape[0]* ofms_shape[1]* ofms_shape[2]))
np.savetxt(ofms_file, ofms, fmt='%i')