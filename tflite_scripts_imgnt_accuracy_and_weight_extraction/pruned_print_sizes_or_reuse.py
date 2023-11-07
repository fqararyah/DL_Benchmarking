import os
from fcntl import F_SETFL
from models_archs import utils
import analysis_utils

def get_layers_op_counts(model_dag):

    layers_num_of_ops = []
    for layer_specs in model_dag:
        if 'type' in layer_specs and layer_specs['type'] in ['s', 'dw', 'pw']:
            layers_ofms_shape = layer_specs['ofms_shape']
            layers_weights_shape = layer_specs['weights_shape']

            layer_num_of_ops = 1
            for i in layers_weights_shape:
                layer_num_of_ops *= i

            layer_num_of_ops *= layers_ofms_shape[1] * layers_ofms_shape[2]

            layers_num_of_ops.append(layer_num_of_ops)

    return layers_num_of_ops

def get_fms_sizes(model_dag):

    fms_sizes = []
    for layer_specs in model_dag:
        if 'type' in layer_specs and layer_specs['type'] in ['s', 'dw', 'pw']:
            fms_sizes.append(layer_specs['ifms_shape'][0] * layer_specs['ifms_shape'][1] * layer_specs['ifms_shape'][2] +
                            layer_specs['ofms_shape'][0] * layer_specs['ofms_shape'][1] * layer_specs['ofms_shape'][2])

    return fms_sizes


def get_weights_sizes(model_dag):

    weights_sizes = []
    for layer_specs in model_dag: 
        if 'type' in layer_specs and layer_specs['type'] in ['s', 'dw', 'pw']:
            weights_shape = layer_specs['weights_shape']
            if layer_specs['type'] in ['s']:
                weights_sizes.append(
                    weights_shape[0] * weights_shape[1] * weights_shape[2] * weights_shape[3])
            elif layer_specs['type'] in ['dw']:
                weights_sizes.append(
                    weights_shape[0] * weights_shape[1] * weights_shape[2])
            elif layer_specs['type'] in ['pw']:
                weights_sizes.append(weights_shape[0] * weights_shape[1])

    return weights_sizes


def print_filters_channels(model_dag):

    for layer_specs in model_dag:
        if 'type' in layer_specs and layer_specs['type'] in ['s', 'dw', 'pw']:
            print(layer_specs['weights_shape'][0]
                * layer_specs['weights_shape'][0])


dag_file = '/media/SSD2TB/fareed/wd/models/codesign/batch1_model_dags/model_{}.tflite.json'


MODEL_NAME = 'mob_v2' #uniform_mobilenetv2_75

utils.set_globals(MODEL_NAME, MODEL_NAME)

model_dag = model_dag = utils.read_model_dag()
base_layers_num_of_ops = sum(get_layers_op_counts(model_dag))
base_weight_sizes = sum(get_weights_sizes(model_dag))

for i in range(1000):
    current_dag_file = dag_file.format(i)
    if os.path.exists(current_dag_file):
        model_dag = model_dag = utils.read_model_dag_file(current_dag_file)
        weight_reuse = []
        activation_reuse = []
        activations_sizes = []
        weight_reuse_and_ops = {}
        activation_reuse_and_ops = {}
        layers_num_of_ops = []

        weight_sizes = sum(get_weights_sizes(model_dag))

        layers_num_of_ops = sum(get_layers_op_counts(model_dag))

        #print(weight_sizes/base_weight_sizes)
        print(layers_num_of_ops/base_layers_num_of_ops)
    else:
        print('DNE')