from inspect import currentframe
import json
from operator import mod
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras.applications.mobilenet_v2 as mob_v2
import tensorflow.keras.applications.mobilenet as mob_v1
import tensorflow.keras.applications as models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import pathlib

from models_archs import utils

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#################################################################################################################
MODEL_NAME = 'mob_v2_v3'

# from generic to specific (for string matching)
ACTIVATION_FUNCTIONS = ['relu', 'relu6']
TFLITE_CONV_OP_NAMES = ['depthwise_conv_2d', 'conv_2d']

PRECISION = 8
np.random.seed(0)

weights_fms_dir = MODEL_NAME
model_arch_dir = './models_archs/models/' + MODEL_NAME + '/'
tflite_models_dir = pathlib.Path("./")
tflite_model_quant_file = tflite_models_dir / \
    (MODEL_NAME + '_' + str(PRECISION) + ".tflite")
#################################################################################################################
interpreter = tf.lite.Interpreter(model_path=str(
    tflite_model_quant_file), experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
ops_details_list = interpreter._get_ops_details()
tensors_details_list = interpreter.get_tensor_details()
# print(interpreter._get_op_details(1))
# print(interpreter._get_op_details(2))
# print(interpreter.get_tensor_details()[3])
# print(interpreter.get_tensor_details()[4])

model_dag = []
tmp_ofms_to_layer_indeices_map = {}

op_index_comp = 0
for op_details in ops_details_list:
    model_dag_entry = {}
    op_name = op_details['op_name'].lower()
    model_dag_entry['name'] = op_name
    op_index = op_details['index']
    print('processing layer:', op_index)
    assert op_index == op_index_comp
    model_dag_entry['id'] = op_index
    op_index_comp += 1
    op_inputs = op_details['inputs']
    op_outputs = op_details['outputs']
    op_inputs = sorted(op_inputs)

    op_ofms_tensor_details = tensors_details_list[op_outputs[0]]
    op_ofms_tensor = interpreter.get_tensor(op_outputs[0])
    op_ofms_tensor = np.squeeze(op_ofms_tensor)
    if op_ofms_tensor.ndim == 3:
        op_ofms_tensor = np.transpose(op_ofms_tensor, (2, 0, 1))
        op_ofms_tensor = np.reshape(op_ofms_tensor, (op_ofms_tensor.size))
        file_name = 'ofms_' + str(op_index) + '_' + op_name + '.txt'
        np.savetxt('./'+weights_fms_dir+'/fms/' + file_name, op_ofms_tensor)

    tmp_ofms_to_layer_indeices_map[op_outputs[0]] = op_index
    model_dag_entry['parents'] = []
    for op_input in op_inputs:
        if op_input in tmp_ofms_to_layer_indeices_map:
            model_dag_entry['parents'].append(
                tmp_ofms_to_layer_indeices_map[op_input])

    if op_name == 'add':
        model_dag_entry['ifms_scales'] = []
        model_dag_entry['ifms_zero_points'] = []
        for op_input in op_inputs:
            op_ifms_tensor_details = tensors_details_list[op_input]
            assert(
                len(op_ifms_tensor_details['quantization_parameters']['scales']) == 1)
            model_dag_entry['ifms_scales'].append(
                float(op_ifms_tensor_details['quantization_parameters']['scales'][0]))
            assert(
                len(op_ifms_tensor_details['quantization_parameters']['zero_points']) == 1)
            model_dag_entry['ifms_zero_points'].append(
                float(op_ifms_tensor_details['quantization_parameters']['zero_points'][0]))

    elif op_name in TFLITE_CONV_OP_NAMES:
        # assuming the op_inputs are of the weights, then the biases, theen the IFMs (based on my observation)
        op_ifms_tensor = interpreter.get_tensor(op_inputs[-1])
        op_ifms_tensor_details = tensors_details_list[op_inputs[-1]]
        op_weights_tensor = interpreter.get_tensor(op_inputs[0])
        op_weights_tensor_details = tensors_details_list[op_inputs[0]]
        op_biases_tensor = interpreter.get_tensor(op_inputs[1])
        op_biases_tensor_details = tensors_details_list[op_inputs[1]]

        op_ifms_tensor = np.squeeze(op_ifms_tensor)
        op_ifms_tensor = np.transpose(op_ifms_tensor, (2, 0, 1))
        op_ifms_tensor = np.reshape(op_ifms_tensor, (op_ifms_tensor.size))
        file_name = 'ifms_' + str(op_index) + '_' + op_name + '.txt'
        np.savetxt('./'+weights_fms_dir+'/fms/' + file_name, op_ifms_tensor)

        op_weights_tensor = np.squeeze(op_weights_tensor)
        if op_weights_tensor.ndim == 4:
            op_weights_tensor = np.transpose(op_weights_tensor, (0, 3, 1, 2))
        elif op_weights_tensor.ndim == 3:
            op_weights_tensor = np.transpose(op_weights_tensor, (2,0,1))
        op_weights_tensor = np.reshape(op_weights_tensor, (op_weights_tensor.size))
        file_name = 'weights_' + str(op_index) + '_' + op_name + '.txt'
        np.savetxt('./'+weights_fms_dir+'/weights/' + file_name, op_weights_tensor)
        
        file_name = 'biases_' + str(op_index) + '_' + op_name + '.txt'
        np.savetxt('./'+weights_fms_dir+'/biases/' + file_name, op_biases_tensor)

        op_weights_shape = [int(i) for i in op_weights_tensor_details['shape']]

        if 'depthwise' in op_name:
            model_dag_entry['type'] = 'dw'
            model_dag_entry['weights_shape'] = [
                op_weights_shape[2], op_weights_shape[0], op_weights_shape[1]]
        elif op_weights_shape[1] == 1 and op_weights_shape[2] == 1:
            model_dag_entry['type'] = 'pw'
            model_dag_entry['weights_shape'] = [
                op_weights_shape[0], op_weights_shape[1]]
        else:
            model_dag_entry['type'] = 's'
            model_dag_entry['weights_shape'] = [
                op_weights_shape[0], op_weights_shape[3], op_weights_shape[1], op_weights_shape[2]]

        model_dag_entry['ifms_shape'] = [int(op_ifms_tensor_details['shape'][3]), int(op_ifms_tensor_details['shape'][1]),
                                         int(op_ifms_tensor_details['shape'][2])]
        model_dag_entry['ofms_shape'] = [int(op_ofms_tensor_details['shape'][3]), int(op_ofms_tensor_details['shape'][1]),
                                         int(op_ofms_tensor_details['shape'][2])]
        model_dag_entry['strides'] = int(
            model_dag_entry['ifms_shape'][-1] / model_dag_entry['ofms_shape'][-1])

        for activation in ACTIVATION_FUNCTIONS:
            if activation in op_ofms_tensor_details['name'].lower():
                model_dag_entry['activation'] = activation
            else:
                model_dag_entry['activation'] = '0'

    model_dag_entry['ofms_scales'] = []
    model_dag_entry['ofms_zero_points'] = []
    assert(
        len(op_ofms_tensor_details['quantization_parameters']['scales']) == 1)
    model_dag_entry['ofms_scales'].append(
        float(op_ofms_tensor_details['quantization_parameters']['scales'][0]))
    assert(
        len(op_ofms_tensor_details['quantization_parameters']['zero_points']) == 1)
    model_dag_entry['ofms_zero_points'].append(
        float(op_ofms_tensor_details['quantization_parameters']['zero_points'][0]))

    model_dag.append(model_dag_entry)

json_object = json.dumps(model_dag)

with open(model_arch_dir + "model_dag.json", "w") as outfile:
    outfile.write(json_object)

exit()
output_details = interpreter.get_output_details()[0]
#################################################################################################################
# prepare image
test_image = '/media/SSD2TB/shared/vedliot_evaluation/D3.3_Accuracy_Evaluation/imagenet/imagenet_val2012/ILSVRC2012_val_00018455.JPEG'
a_test_image = load_img(test_image, target_size=(224, 224))
numpy_image = img_to_array(a_test_image, dtype=np.uint8)
image_batch = np.expand_dims(numpy_image, axis=0)
#################################################################################################################
# invoke mode
interpreter.set_tensor(input_details["index"], image_batch)
interpreter.invoke()

tensor_details = interpreter.get_tensor_details()
#################################################################################################################

# this relys on the observation that tfl.quantize is the first fms tensor
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
    #print(tensor_name, interpreter.get_tensor(t['index']).shape)
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
        #print(tensor_name, t['index'], original_shape)
        if original_tensor.ndim == 1:
            if last_tensor_key == '':
                continue

            file_name = last_tensor_key + '_' + \
                str(weights_counts[last_tensor_key] - 1) + '_biases'
            # if MODEL_NAME == 'mnas' and os.path.exists('./'+weights_fms_dir+'/biases/' +
            #         file_name + '.txt'):
            #     file_name = last_tensor_key + '_' + str(weights_counts[last_tensor_key]) + '_biases'
            np.savetxt('./'+weights_fms_dir+'/biases/' +
                       file_name + '.txt', current_tensor, fmt='%i')
            np.savetxt('./'+weights_fms_dir+'/biases/' + file_name +
                       '_scales.txt', t['quantization_parameters']['scales'])
            np.savetxt('./'+weights_fms_dir+'/biases/' + file_name + '_zero_points.txt',
                       t['quantization_parameters']['zero_points'], fmt='%i')
        else:
            if 'conv2d;' in tensor_name:
                tensor_name_postfix += 'conv2d'
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
                layers_activations.append(ACTIVATION_FUNCTION[MODEL_NAME])
                layers_strides.append(1)
                if original_shape[0] == 1:
                    conv_type = 'dw'
                elif original_shape[1] == 1 and original_shape[2] == 1:
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

        file_name = last_tensor_key + '_' + \
            str(fms_counts[last_tensor_key] - 1)
        if last_tensor_key not in tensor_name_postfix:
            file_name += '_' + tensor_name_postfix + \
                '_' + str(internal_layers_count)
            layers_execution_sequence.append(
                tensor_name_postfix + '_' + str(internal_layers_count))
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
                    and layers_inputs_dims[-1][2] != layers_inputs_dims[-2][2]:
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
    i = 0
    for layer_activation in layers_activations:
        if len(layers_outputs_dims[i]) == 1:
            layer_activation = ACTIVATION_FUNCTION[MODEL_NAME]
        f.write(layer_activation + '\n')
        i += 1

with open(model_arch_dir + 'layers_strides.txt', 'w') as f:
    for layer_strides in layers_strides:
        f.write(str(layer_strides) + '\n')

with open(model_arch_dir + 'skip_connections_indices.txt', 'w') as f:
    for skip_connection_index in skip_connection_indices:
        f.write(str(skip_connection_index) + '\n')

with open(model_arch_dir + 'layers_execution_sequence.txt', 'w') as f:
    for layer in layers_execution_sequence:
        f.write(layer + '\n')
