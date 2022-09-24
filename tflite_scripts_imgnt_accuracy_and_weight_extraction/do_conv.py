import numpy as np

weights_file = './weights/weights_3_pw.txt'
ifms_file = './fms/fms_3_32_112_112.txt'
ofms_file = './scratch_out/ofms.txt'

weights_shape = (16, 32, 1, 1)
ifms_shape = (32, 112, 112)

conv_strides = 1

weights = np.loadtxt(weights_file).astype(np.int8)
weights = np.reshape(weights, weights_shape)

ifms = np.loadtxt(ifms_file).astype(np.int8)
ifms = np.reshape(ifms, ifms_shape)
ifms = np.pad(ifms, ((0,0),(0,1),(0,1)), mode='constant')

ofms_shape = (weights_shape[0], int(ifms_shape[1] /
                                    conv_strides), int(ifms_shape[2]/conv_strides))
ofms = np.zeros(ofms_shape).astype(np.int32)

filter_height = weights_shape[2]
filter_width = weights_shape[3]
print('>>>', weights[0])
print('>>>', ifms[:, 0*conv_strides:0*conv_strides +
                                              filter_height, 0*conv_strides:0*conv_strides + filter_width])
print('>>>', weights[0].astype(np.int32)  * ifms[:, 0*conv_strides:0*conv_strides +
                                              filter_height, 0*conv_strides:0*conv_strides + filter_width])
for i in range(weights_shape[0]):
    for j in range(ofms_shape[1]):
        for k in range(ofms_shape[2]):
            ofms[i][j][k] = (np.sum(weights[i].astype(np.float32) * ifms[:, j*conv_strides:j*conv_strides +
                                              filter_height, k*conv_strides:k*conv_strides + filter_width]) - 2630) #* 0.000683013 + 6

print(np.max(ofms[0]))
print(np.max(ofms[1]))
print(np.max(ofms[2]))
print(np.max(ofms[3]))
print(np.max(ofms))
ofms = ofms.reshape((ofms_shape[0]* ofms_shape[1]* ofms_shape[2]))
np.savetxt(ofms_file, ofms, fmt='%i')