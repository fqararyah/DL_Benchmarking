import numpy as np
import os

ifms_folder = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/fpga_out_1'

weights_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/weights/fc_weights.txt'
weight_sums_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/weights/fc_weight_sums.txt'
weights_scales_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/' +\
    'weights/fc_weight_scales.txt'
biases_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/weights/fc_biases.txt'
biases_scales_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/' +\
    'weights/fc_biases_scales.txt'

ofms_file = './fc_out/{}.txt'

image_names = []
ifms_files = []


def locate_ifms(path):
    for (root, dirs, file) in os.walk(path):
        for f in file:
            if '.txt' in f:
                ifms_files.append(os.path.abspath(os.path.join(path, f)))
                image_names.append(f.split('.')[0])


locate_ifms(ifms_folder)

fc_ifm_scale = 0.020379824563860893
fc_ifm_zero_point = -128
#fc_ofm_scale = 0.07442259788513184
#fc_ofm_zero_point = -59
i = 0

weights = np.loadtxt(weights_file).astype(np.int64)
weight_scales = np.loadtxt(weights_scales_file)
weights = np.reshape(weights, (1000, 1280))
weight_sums = np.loadtxt(weight_sums_file).astype(np.int32)
#np.savetxt(weight_sums_file,weight_sums, '%i')
biases = np.loadtxt(biases_file).astype(np.int32)
baises_scale = np.loadtxt(biases_scales_file)
scaled_biases = biases * baises_scale

print(weight_scales)
print(baises_scale)


for ifms_file in ifms_files:
    #print(ifms_file)
    ifms = np.loadtxt(ifms_file).astype(np.int64)
    scaled_ifms = (ifms - fc_ifm_zero_point)

    ofms = np.dot(weights, ifms) + \
        (- weight_sums * fc_ifm_zero_point) + scaled_biases / (weight_scales * fc_ifm_scale) 
    #ofms = fc_ofm_zero_point + ( (np.dot(weights, ifms) - weight_sums * fc_ifm_zero_point ) * 
    # (weight_scales *fc_ifm_scale) + scaled_biases) / fc_ofm_scale

    np.savetxt(ofms_file.format(image_names[i]), ofms)
    i += 1
