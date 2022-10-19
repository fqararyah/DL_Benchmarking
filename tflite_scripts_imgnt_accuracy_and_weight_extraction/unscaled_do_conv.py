from ctypes import util
import numpy as np
from models_archs import utils


layer_index = 3
layers_types = utils.read_layers_types()
layers_weights_shape = utils.read_layers_weight_shapes(layers_types)
layers_ifms_shape = utils.read_layers_input_shapes()
layers_strides = utils.read_layers_strides()
layer_type = layers_types[layer_index]
weights_file = './weights/weights_{}_{}.txt'.format(layer_index, layer_type)
ifms_file = './fms/fms_{}_{}_{}_{}.txt'.format(layer_index if layer_index > 0 else 1, layers_ifms_shape[layer_index].depth, layers_ifms_shape[layer_index].height,\
    layers_ifms_shape[layer_index].width)
ofms_file = './scratch_out/ofms_{}_un.txt'.format(layer_index)
ifms_zero_points_file = './fms/fms_{}_zero_points.txt'.format(layer_index if layer_index > 0 else 1)
bias_file = './weights/weights_{}_biases.txt'.format(layer_index)

layers_ifms_zero_point = {layer_index: np.loadtxt(ifms_zero_points_file).astype(np.int32)}
layers_bias = {layer_index: np.loadtxt(bias_file).astype(np.int32)}
#layers_ifms_zero_point = {0: 128, 3: 128, 6: 128, 4: 6, 5:128}#{layer_index: np.loadtxt(ifms_zero_points_file).astype(np.int8)}
#layers_bias = {0: [61864]*32, 3: [-2630]*32, 6: [32910]*32, 4: [2650]*32}#{layer_index: np.loadtxt(bias_file).astype(np.int32)}
print(layers_ifms_zero_point[layer_index])
conv_strides = layers_strides[layer_index]

weights = np.loadtxt(weights_file).astype(np.int8)
weights = np.reshape(weights,(layers_weights_shape[layer_index].num_of_filters, layers_weights_shape[layer_index].depth, layers_weights_shape[layer_index].height,\
     layers_weights_shape[layer_index].width))

ifms = np.loadtxt(ifms_file).astype(np.int8)
ifms = np.reshape(ifms, (layers_ifms_shape[layer_index].depth, layers_ifms_shape[layer_index].height, layers_ifms_shape[layer_index].width) )

filter_height =layers_weights_shape[layer_index].height
filter_width =layers_weights_shape[layer_index].width
padding_val = int( (filter_height - 1) / 2)
if layer_type != 'pw':
    if conv_strides == 1:
        ifms = np.pad(ifms, ((0,0),(padding_val,padding_val),(padding_val,padding_val)), mode='constant', constant_values = layers_ifms_zero_point[layer_index])
    elif conv_strides == 2:
        ifms = np.pad(ifms, ((0,0),(0,padding_val),(0,padding_val)), mode='constant',  constant_values = layers_ifms_zero_point[layer_index])


ofms_shape = (layers_weights_shape[layer_index].num_of_filters, int(layers_ifms_shape[layer_index].height /
                                    conv_strides), int(layers_ifms_shape[layer_index].width/conv_strides))
ofms = np.zeros(ofms_shape).astype(np.int32)

filter_height =layers_weights_shape[layer_index].height
filter_width =layers_weights_shape[layer_index].width
# print('>>>', weights[0])
# print('>>>', ifms[:, 0*conv_strides:0*conv_strides +
#                                               filter_height, 0*conv_strides:0*conv_strides + filter_width])
# print('>>>', weights[0].astype(np.int32)  * ifms[:, 0*conv_strides:0*conv_strides +
#                                               filter_height, 0*conv_strides:0*conv_strides + filter_width])
for i in range(layers_weights_shape[layer_index].num_of_filters):
    for j in range(ofms_shape[1]):
        for k in range(ofms_shape[2]):
            # ofms[i][j][k] = (np.sum(weights[i].astype(np.float32) * ( -layers_ifms_zero_point[layer_index] + \
            #     ifms[:, j*conv_strides:j*conv_strides + \
            #         filter_height, k*conv_strides:k*conv_strides + filter_width]) ) + layers_bias[layer_index][i])

            ofms[i][j][k] = np.sum(weights[i].astype(np.float32) * \
                ifms[:, j*conv_strides:j*conv_strides + \
                    filter_height, k*conv_strides:k*conv_strides + filter_width])

print(weights[0,:, 0, 0])
print(ifms[:, 0, 0])
#exit()
print(np.sum(weights[0,0, 0, 0].astype(np.float32) * \
                ifms[0, 0, 0]))
print(np.sum(weights[0,0:2, 0, 0].astype(np.float32) * \
                ifms[0:2, 0, 0]))
print(np.sum(weights[0,0:3, 0, 0].astype(np.float32) * \
                ifms[0:3, 0, 0]))

print(np.sum(weights[0,:, 0, 0].astype(np.float32) * \
                ifms[:, 0, 0]))

print(np.sum(weights[0].astype(np.float32) * \
                ifms[:, 0:3, 0:3]))
# print(np.max(ofms[0]))
# print(np.max(ofms[1]))
# print(np.max(ofms[2]))
# print(np.max(ofms[3]))
# print(np.max(ofms))
ofms = ofms.reshape((ofms_shape[0]* ofms_shape[1]* ofms_shape[2]))
np.savetxt(ofms_file, ofms, fmt='%i')