import numpy as np

layer_index = 3
layers_weights_shape = {3: (16, 32, 1, 1), 6: (24, 96, 1, 1)}
layers_ifms_shape = {3: (32, 112, 112), 6: (96, 56, 56)}
weights_file = './weights/weights_{}_pw.txt'.format(layer_index)
ifms_file = './fms/fms_{}_{}_{}_{}.txt'.format(layer_index, layers_ifms_shape[layer_index][0], layers_ifms_shape[layer_index][1],\
    layers_ifms_shape[layer_index][2])
ofms_file = './scratch_out/ofms_{}.txt'.format(layer_index)

layers_zero_point_ifms = {3: 128, 6: 128}
layers_bias = {3: -2630, 6: 32910}

layers_weights_shape = {3: (16, 32, 1, 1), 6: (24, 96, 1, 1)}
layers_ifms_shape = {3: (32, 112, 112), 6: (96, 56, 56)}

conv_strides = 1

weights = np.loadtxt(weights_file).astype(np.int8)
weights = np.reshape(weights,layers_weights_shape[layer_index])

ifms = np.loadtxt(ifms_file).astype(np.int8)
ifms = np.reshape(ifms, layers_ifms_shape[layer_index])
ifms = np.pad(ifms, ((0,0),(0,1),(0,1)), mode='constant')

ofms_shape = (layers_weights_shape[layer_index][0], int(layers_ifms_shape[layer_index][1] /
                                    conv_strides), int(layers_ifms_shape[layer_index][2]/conv_strides))
ofms = np.zeros(ofms_shape).astype(np.int32)

filter_height =layers_weights_shape[layer_index][2]
filter_width =layers_weights_shape[layer_index][3]
print('>>>', weights[0])
print('>>>', ifms[:, 0*conv_strides:0*conv_strides +
                                              filter_height, 0*conv_strides:0*conv_strides + filter_width])
print('>>>', weights[0].astype(np.int32)  * ifms[:, 0*conv_strides:0*conv_strides +
                                              filter_height, 0*conv_strides:0*conv_strides + filter_width])
for i in range(layers_weights_shape[layer_index][0]):
    for j in range(ofms_shape[1]):
        for k in range(ofms_shape[2]):
            ofms[i][j][k] = (np.sum(weights[i].astype(np.float32) * (layers_zero_point_ifms[layer_index] + \
                ifms[:, j*conv_strides:j*conv_strides + \
                    filter_height, k*conv_strides:k*conv_strides + filter_width]) ) + layers_bias[layer_index])

print(np.max(ofms[0]))
print(np.max(ofms[1]))
print(np.max(ofms[2]))
print(np.max(ofms[3]))
print(np.max(ofms))
ofms = ofms.reshape((ofms_shape[0]* ofms_shape[1]* ofms_shape[2]))
np.savetxt(ofms_file, ofms, fmt='%i')