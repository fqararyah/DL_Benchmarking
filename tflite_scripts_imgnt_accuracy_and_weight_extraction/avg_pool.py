import numpy as np

ifms_file = './mob_v2/fms/fms_conv2d_{}_{}_{}_{}.txt'.format(52, 1280, 7, 7)
ofms_file = './scratch_out/ofms_avgpool_{}_ref.txt'.format(1)


ifms = np.loadtxt(ifms_file).astype(np.int8)
ifms = np.reshape(ifms, (1280, 7, 7))

ofms = np.sum(ifms, axis=(1,2))

quantized_ofms =  ((0.0235294122248888 / 0.020379824563860893 ) * ( 128 + (ofms) / ( 49) )\
    - 128 ).astype(np.int8)

print(ofms.shape,ofms)
print(quantized_ofms.shape,quantized_ofms)

np.savetxt(ofms_file, quantized_ofms, fmt='%i')
