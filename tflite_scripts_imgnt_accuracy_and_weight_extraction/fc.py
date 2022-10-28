import numpy as np

ifms_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/weights/weights_52_1_biases.txt'
weights_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/weights/fc_weights.txt'
weights_scales_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/'+\
    'weights/fc_weight_scales.txt'
biases_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/weights/fc_biases.txt'
biases_scales_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/'+\
    'weights/fc_biases_scales.txt'
fc_biases_scales_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/'+\
    'tflite_scripts_imgnt_accuracy_and_weight_extraction/weights/fc_biases_scales.txt'
ofms_file = './scratch_out/ofms_fc_{}_ref.txt'.format(1)


ifms = np.loadtxt(ifms_file).astype(np.int8)
scaled_ifms = 0.020379824563860893 * (ifms + 128)
#print(scaled_ifms)
weights = np.loadtxt(weights_file).astype(np.int8)
biases = np.loadtxt(biases_file).astype(np.int32)

scaled_weights = np.loadtxt(weights_scales_file) * weights
#print(weights[0])
scaled_weights = np.reshape(scaled_weights, (1000, 1280))
print(scaled_weights.shape)
print(ifms.shape)
ofms = -59 + (np.dot(scaled_weights, scaled_ifms) + biases * np.loadtxt(biases_scales_file)) / 0.07442259788513184
ofms = ofms.astype(np.int8)
print(ofms.shape, ofms[0:10])


