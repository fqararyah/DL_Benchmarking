
import os
import numpy as np
import json

fc_out_folder = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/fc_out'
predictions_file = './predictions.json'

fc_out_files = []
file_names = []


def locate_ifms(path):
    for (root, dirs, file) in os.walk(path):
        for f in file:
            if '.txt' in f:
                fc_out_files.append(os.path.abspath(os.path.join(path, f)))
                file_names.append(f)


locate_ifms(fc_out_folder)

prediction_dict_list = []
for i in range(len(fc_out_files)):
    current_fc_out = np.loadtxt(fc_out_files[i])
    indices = np.argsort(current_fc_out)[-5:]
    indices = np.flip(indices)
    prediction_dict = {"dets": indices.tolist(), "image": file_names[i].replace('.txt', '.JPEG')}
    prediction_dict_list.append(prediction_dict)

json_object = json.dumps(prediction_dict_list)
with open(predictions_file, "w") as outfile:
    outfile.write(json_object)