import numpy as np
import os

ifms_folder = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/fpga_out_1'
weights_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/weights/fc_weights.txt'
weights_scales_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/' +\
    'weights/fc_weight_scales.txt'
biases_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/weights/fc_biases.txt'
biases_scales_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/' +\
    'weights/fc_biases_scales.txt'
fc_biases_scales_file = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/' +\
    'tflite_scripts_imgnt_accuracy_and_weight_extraction/weights/fc_biases_scales.txt'
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

i = 0
for ifms_file in ifms_files:
    print(ifms_file)
    ifms = np.loadtxt(ifms_file).astype(np.int8)
    scaled_ifms = 0.020379824563860893 * (ifms + 128)
    # print(scaled_ifms)
    weights = np.loadtxt(weights_file).astype(np.int8)
    biases = np.loadtxt(biases_file).astype(np.int32)

    scaled_weights = np.loadtxt(weights_scales_file) * weights
    # print(weights[0])
    scaled_weights = np.reshape(scaled_weights, (1000, 1280))
    # print(scaled_weights.shape)
    # print(ifms.shape)
    ofms = -59 + (np.dot(scaled_weights, scaled_ifms) + biases *
                  np.loadtxt(biases_scales_file)) / 0.07442259788513184
    ofms = ofms.astype(np.int8)
    np.savetxt(ofms_file.format(image_names[i]), ofms, fmt='%i')
    i += 1
    #print(ofms.shape, ofms[0:10])
