import numpy as np

ifms_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/weights/weights_52_1_biases.txt'
weights_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction' +\
    '/non_conv_layers/layer_2_1000_1280.txt'
biases_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction' +\
    '/non_conv_layers/layer_3_1000.txt'
ofms_file = './scratch_out/ofms_fc_{}_ref.txt'.format(1)


ifms = np.loadtxt(ifms_file).astype(np.int8)
scaled_ifms = 0.020379824563860893 * (ifms + 128)
#print(scaled_ifms)
weights = np.loadtxt(weights_file).astype(np.int8)
biases = np.loadtxt(biases_file).astype(np.int32)

weights = 0.0018739686347544193 * weights
#print(weights[0])
weights = np.reshape(weights, (1000, 1280))
#print(biases)
ofms = -59 + (np.dot(weights, scaled_ifms) + biases) / 0.07442259788513184
ofms = ofms.astype(np.int8)
print(ofms.shape, ofms[0:10])


