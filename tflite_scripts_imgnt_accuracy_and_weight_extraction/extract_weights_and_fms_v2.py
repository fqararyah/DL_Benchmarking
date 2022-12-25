from inspect import currentframe
import json
from operator import mod
from sklearn import utils
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras.applications.mobilenet_v2 as mob_v2
import tensorflow.keras.applications as models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import pathlib

from models_archs import utils

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_NAME = 'mob_v2'
ACTIVATION_FUNCTION = 'relu6'
PRECISION = 8
NUM_OF_CLASSES = 1000
np.random.seed(0)

weights_fms_dir = MODEL_NAME
model_arch_dir = './models_archs/models/' + MODEL_NAME + '/'
tflite_models_dir = pathlib.Path("./")
tflite_model_quant_file = tflite_models_dir / \
    (MODEL_NAME + '_' + str(PRECISION) + ".tflite")

interpreter = tf.lite.Interpreter(model_path=str(
    tflite_model_quant_file), experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# prepare image
test_image = '/media/SSD2TB/shared/vedliot_evaluation/D3.3_Accuracy_Evaluation/imagenet/imagenet_val2012/ILSVRC2012_val_00018455.JPEG'
a_test_image = load_img(test_image, target_size=(224, 224))
numpy_image = img_to_array(a_test_image, dtype=np.uint8)
image_batch = np.expand_dims(numpy_image, axis=0)

# invoke mode
interpreter.set_tensor(input_details["index"], image_batch)
interpreter.invoke()

tensor_details = interpreter.get_tensor_details()

splitting_layer_name = 'tfl.quantize'
current_tensors_type = 'weights'
weights_counts = {'conv2d': 0, 'matmul': 0}
fms_counts = {'conv2d': 1, 'matmul': 0}
skip_connection_layers_types = ['mul', 'add']
skip_connection_indices = []
last_tensor_key = ''
layers_weights_dims = []
layers_inputs_dims = [[3, 224, 224]]
layers_outputs_dims = []
layers_types = []
layers_activations = []
layers_strides = []
secondary_layers_types = []
layers_execution_sequence = []
fms_count = 0
layer_count = 0
internal_layers_count = 1
fc_biases_found = False
initial_ifms_file_name = 'conv2d_0'
for t in interpreter.get_tensor_details():
    tensor_name = t['name'].lower()
    tensor_name = tensor_name.replace('depthwise', 'conv2d')
    tensor_name_postfix = tensor_name.split('/')[-1]
    if tensor_name == splitting_layer_name:
        current_tensors_type = 'fms'

    original_tensor = interpreter.get_tensor(t['index'])  # .astype(np.int8)
    original_shape = original_tensor.shape
    current_tensor = np.squeeze(original_tensor)
    if current_tensor.ndim == 3:
        current_tensor = np.transpose(current_tensor, (2, 0, 1))
    elif current_tensor.ndim == 4:
        current_tensor = np.transpose(current_tensor, (0, 3, 1, 2))

    current_tensor_dims = current_tensor.shape
    current_tensor_shape_str_rep = str([i for i in current_tensor.shape]).replace(
        ' ', '').replace('[', '').replace(']', '').replace(',', '_')

    current_tensor = np.reshape(current_tensor, (current_tensor.size))

    if current_tensors_type == 'weights':
        if original_tensor.ndim == 1:
            if last_tensor_key == '':
                continue
            file_name = last_tensor_key + '_' + str(weights_counts[last_tensor_key] - 1) + '_biases'
            np.savetxt('./'+weights_fms_dir+'/biases/' +
                    file_name + '.txt', current_tensor, fmt='%i')
            np.savetxt('./'+weights_fms_dir+'/biases/' + file_name +
                    '_scales.txt', t['quantization_parameters']['scales'])
            np.savetxt('./'+weights_fms_dir+'/biases/' + file_name + '_zero_points.txt',
                    t['quantization_parameters']['zero_points'], fmt='%i')
        else:
            for key, val in weights_counts.items():
                if key in tensor_name_postfix:
                    weights_counts[key] += 1
                    last_tensor_key = key
                    break
            if last_tensor_key == '':
                continue

            weights_file_name_postifix = ''

            file_name = last_tensor_key + '_' + \
                str(weights_counts[last_tensor_key] - 1)

            if last_tensor_key == 'conv2d' and 'conv2d' in tensor_name_postfix:
                layers_activations.append(ACTIVATION_FUNCTION)
                layers_strides.append(1)
                if original_shape[0] == 1:
                    conv_type = 'dw'
                elif original_shape[1]== 1 and original_shape[2] == 1:
                    conv_type = 'pw'
                    if current_tensor_dims[1] > current_tensor_dims[0]:
                        layers_activations[-1] = '0'
                else:
                    conv_type = 's'
                
                layers_types.append(conv_type)
                layers_weights_dims.append(current_tensor_dims)

                weights_file_name_postifix += '_' + conv_type + '_weights'

            np.savetxt('./'+weights_fms_dir+'/weights/' +
                    file_name + weights_file_name_postifix + '.txt', current_tensor, fmt='%i')
            np.savetxt('./'+weights_fms_dir+'/weights/' + file_name +
                    '_scales.txt', t['quantization_parameters']['scales'])
            np.savetxt('./'+weights_fms_dir+'/weights/' + file_name + '_zero_points.txt',
                    t['quantization_parameters']['zero_points'], fmt='%i')
    else:
        if 'conv2d;' in tensor_name:
            tensor_name_postfix += 'conv2d'
        for key, val in fms_counts.items():
            if key in tensor_name_postfix:
                fms_counts[key] += 1
                last_tensor_key = key
                break
        
        file_name = last_tensor_key + '_' + str(fms_counts[last_tensor_key] - 1)
        if last_tensor_key not in tensor_name_postfix:
            file_name += '_' + tensor_name_postfix + '_' + str(internal_layers_count)
            layers_execution_sequence.append(tensor_name_postfix + '_' + str(internal_layers_count))
            internal_layers_count += 1
            if tensor_name_postfix in skip_connection_layers_types:
                skip_connection_indices.append(fms_counts[last_tensor_key] - 2)
            if tensor_name_postfix not in secondary_layers_types and tensor_name_postfix.isidentifier():
                secondary_layers_types.append(tensor_name_postfix)
        else:
            layers_execution_sequence.append('conv2d')
            internal_layers_count = 1
            layers_inputs_dims.append(current_tensor_dims)
            layers_outputs_dims.append(current_tensor_dims)
            if len(layers_inputs_dims[-1]) == 3 and len(layers_inputs_dims[-2]) == 3 \
                and  layers_inputs_dims[-1][2] != layers_inputs_dims[-2][2]:
                layers_strides[fms_counts[last_tensor_key] - 2] = 2

        if fms_counts['conv2d'] == 1:
            file_name = initial_ifms_file_name
        
        #print(fms_counts['conv2d'], file_name + '_' + current_tensor_shape_str_rep)
        np.savetxt('./'+weights_fms_dir+'/fms/fms_' + file_name + '_' +
                   current_tensor_shape_str_rep + '.txt', current_tensor, fmt='%i')
        np.savetxt('./'+weights_fms_dir+'/fms/fms_' + file_name + '_scales.txt',
                   t['quantization_parameters']['scales'])
        np.savetxt('./'+weights_fms_dir+'/fms/fms_' + file_name + '_zero_points.txt',
                   t['quantization_parameters']['zero_points'], fmt='%i')

with open(model_arch_dir + 'layers_weights.txt', 'w') as f:
    for shape in layers_weights_dims:
        for i in range(len(shape) - 1):
            f.write(str(shape[i]) + 'x')
        f.write(str(shape[-1]) + '\n')

with open(model_arch_dir + 'layers_types.txt', 'w') as f:
    for layer_type in layers_types:
        f.write(layer_type + '\n')

with open(model_arch_dir + 'secondary_layers_types.txt', 'w') as f:
    for layer_type in secondary_layers_types:
        f.write(layer_type + '\n')

with open(model_arch_dir + 'layers_inputs.txt', 'w') as f:
    for layer_input_dims in layers_inputs_dims:
        for i in range(len(layer_input_dims) - 1):
            f.write(str(layer_input_dims[i]) + 'x')
        f.write(str(layer_input_dims[-1]) + '\n')

with open(model_arch_dir + 'layers_outputs.txt', 'w') as f:
    for layer_output_dims in layers_outputs_dims:
        for i in range(len(layer_output_dims) - 1):
            f.write(str(layer_output_dims[i]) + 'x')
        f.write(str(layer_output_dims[-1]) + '\n')

with open(model_arch_dir + 'layers_activations.txt', 'w') as f:
    for layer_activation in layers_activations:
        f.write(layer_activation + '\n')

with open(model_arch_dir + 'layers_strides.txt', 'w') as f:
    for layer_strides in layers_strides:
        f.write(str(layer_strides) + '\n')

with open(model_arch_dir + 'skip_connections_indices.txt', 'w') as f:
    for skip_connection_index in skip_connection_indices:
        f.write(str(skip_connection_index) + '\n')

with open(model_arch_dir + 'layers_execution_sequence.txt', 'w') as f:
    for layer in layers_execution_sequence:
        f.write(layer + '\n')