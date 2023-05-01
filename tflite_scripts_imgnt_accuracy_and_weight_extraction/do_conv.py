from ctypes import util
import numpy as np
from models_archs import utils


utils.NET_PREFIX = 'resnet50'
utils.set_globals(utils.NET_PREFIX, utils.NET_PREFIX)

model_dag = utils.read_model_dag()

layer_index = 40

layer_specs = model_dag[layer_index]

weights_file = './{}/weights/weights_{}.txt'.format(
    utils.NET_PREFIX, layer_index)
ifms_file = './{}/fms/ifms_{}.txt'.format(utils.NET_PREFIX, layer_index)
ofms_file = './scratch_out/ofms_{}_ref.txt'.format(layer_index)

bias_file = './{}/biases/biases_{}.txt'.format(utils.NET_PREFIX, layer_index)
weights_scale_file = './{}/weights/weights_{}_scales.txt'.format(
    utils.NET_PREFIX, layer_index)

layers_ifms_zero_point = layer_specs['ifms_zero_points']
layers_bias = {layer_index: np.loadtxt(bias_file).astype(np.int32)}
layers_scale_ifms = layer_specs['ifms_scales']
# {0: [0.0095633129], 3: [0.02902807], 6: [0.01043679], 4: [0.00100364], 5: [0.00320544]}
layers_scale_weights = {layer_index: np.loadtxt(weights_scale_file)}
layers_scale_ofms = layer_specs['ofms_scales']
layers_ofms_zero_point = layer_specs['ofms_zero_points']

# print(layers_ifms_zero_point[layer_index])
# print(layers_ofms_zero_point[layer_index])
# print(layers_bias[layer_index])
# print(layers_scale_ifms[layer_index])
# print(layers_scale_weights[layer_index])
# print(layers_scale_ofms[layer_index])
# layers_ofms_zero_point
conv_strides = layer_specs['strides']

layer_type = layer_specs['type']
layers_weights_shape = layer_specs['weights_shape']
layers_ifms_shape = layer_specs['ifms_shape']
ofms_shape = layer_specs['ofms_shape']
layer_activation = layer_specs['activation']

weights = np.loadtxt(weights_file).astype(np.int8)
if layer_type == 'pw':
    weights = np.reshape(
        weights, (layers_weights_shape[0], layers_weights_shape[1]))
else:
    weights = np.reshape(weights, (layers_weights_shape[0], layers_weights_shape[1], layers_weights_shape[2],
                                   layers_weights_shape[3]))

ifms = np.loadtxt(ifms_file).astype(np.int8)
ifms = np.reshape(ifms, (layers_ifms_shape[0], layers_ifms_shape[1], layers_ifms_shape[2]))

ofms = np.zeros(ofms_shape).astype(np.int8)

if layer_specs['type'] != 'pw':
    filter_height = layers_weights_shape[2]
    filter_width = layers_weights_shape[3]
else:
    filter_height = 1
    filter_width = 1
padding_val = int((filter_height - 1) / 2)
# print(layers_ifms_zero_point[layer_index])

if layer_type != 'pw':
    if conv_strides == 1:
        ifms = np.pad(ifms, ((0, 0), (padding_val, padding_val), (padding_val, padding_val)),
                      mode='constant', constant_values=layers_ifms_zero_point)
    elif conv_strides == 2:
        ifms = np.pad(ifms, ((0, 0), (0, padding_val), (0, padding_val)),
                      mode='constant',  constant_values=layers_ifms_zero_point)

def conv():
    for i in range(layers_weights_shape[0]):
        for j in range(ofms_shape[1]):
            for k in range(ofms_shape[2]):
                tmp = np.sum(weights[i].astype(np.int32)) * -layers_ifms_zero_point + \
                    np.sum(weights[i].astype(np.int32) * (ifms[:, j*conv_strides:j*conv_strides + filter_height,
                                                               k*conv_strides:k*conv_strides + filter_width])) + layers_bias[layer_index][i]
                if i == 0 and j == 0 and k == 0:
                    print(filter_height)
                    print(filter_width)
                    print(np.sum(weights[i].astype(np.int32) * (ifms[:, j*conv_strides:j*conv_strides + filter_height,
                                                                                 k*conv_strides:k*conv_strides + filter_width])))
                    # print(weights[i, 31, :])
                tmp = tmp * layers_scale_ifms * \
                    layers_scale_weights[layer_index][i]
                if layer_activation == 'RELU6':
                    tmp = min(max(tmp, 0), 6)
                tmp = tmp / \
                    layers_scale_ofms + \
                    layers_ofms_zero_point
                if tmp > 0:
                    tmp = int(tmp + 0.5)
                else:
                    tmp = int(tmp - 0.5)
                ofms[i][j][k] = tmp


def dw_conv():
    for i in range(layers_weights_shape[layer_index].num_of_filters):
        for j in range(ofms_shape[1]):
            for k in range(ofms_shape[2]):
                tmp = np.sum(weights[i].astype(np.float32)) * - layers_ifms_zero_point[layer_index] + \
                    np.sum(weights[i].astype(np.float32) * ifms[i, j*conv_strides:j*conv_strides +
                                                                filter_height, k*conv_strides:k*conv_strides + filter_width]) + layers_bias[layer_index][i]

                if layer_index == 7 and i == 11 and j == 3:
                    scale_w_i_o = int(0.017232311889529228 * (2**32))
                    tmp = ((int(tmp * scale_w_i_o + (2**32)) >> 32) - 128)
                    print(tmp)
                else:
                    tmp = tmp * \
                        layers_scale_ifms[layer_index] * \
                        layers_scale_weights[layer_index][i]
                    if layer_activation == 'RELU6':
                        tmp = min(max(tmp, 0), 6)
                    tmp = tmp / \
                        layers_scale_ofms[layer_index] + \
                        layers_ofms_zero_point[layer_index]
                    if tmp > 0:
                        tmp = int(tmp + 0.5)
                    else:
                        tmp = int(tmp - 0.5)
                ofms[i][j][k] = tmp

res = 0
if layer_type == 'pw':
    for k in range(layers_weights_shape[1]):
        res += ifms[k][0][2] * weights[0][k].astype(np.int32)
else:
    for l in range(3):
        for m in range(3):
            for k in range(layers_weights_shape[1]):
                res += ifms[k][l][m] * weights[0][k][l][m].astype(np.int32)
                #print(ifms[k][l][m], '*', weights[0][k][l][m])
print(res)

if layer_type == 'dw':
    dw_conv()
else:
    conv()
ofms = ofms.reshape((ofms_shape[0] * ofms_shape[1] * ofms_shape[2]))
np.savetxt(ofms_file, ofms, fmt='%i')
