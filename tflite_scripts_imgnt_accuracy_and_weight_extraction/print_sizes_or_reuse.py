from fcntl import F_SETFL
from models_archs import utils
import analysis_utils

MODEL_NAME = 'mob_v1'  # uniform_mobilenetv2_75

utils.set_globals(MODEL_NAME, MODEL_NAME)

model_dag = model_dag = utils.read_model_dag()

weight_reuse = []
activation_reuse = []
activations_sizes = []
weight_reuse_and_ops = {}
activation_reuse_and_ops = {}
layers_num_of_ops = []

activation_resuse_thresholds = {9: 0, 16: 0, 24: 0, 25: 0, 32: 0, 40: 0, 48: 0, 49: 0, 64: 0, 72: 0, 80: 0, 96: 0, 120: 0, 128: 0,
                                144: 0, 160: 0, 192: 0, 240: 0, 256: 0, 288: 0, 320: 0, 348: 0, 480: 0, 512: 0, 576: 0, 960: 0, 1000: 0, 1152: 0, 1280: 0}

def get_layers_overheads(model_dag, layers_num_of_ops, parallel_f, parallel_h, parallel_w):    

    conv_layer_index = 0
    layers_overheads = []
    for layer_specs in model_dag:
        if 'type' in layer_specs and layer_specs['type'] in ['s', 'dw', 'pw']:
            layers_ofms_shape = layer_specs['ofms_shape']
            layers_weights_shape = layer_specs['weights_shape']
            layer_num_filters = layers_weights_shape[0]
            layer_ofms_height = layers_ofms_shape[1] 
            layer_ofms_width = layers_ofms_shape[2]
            overhead_f = 0
            overhead_h = 0
            overhead_w = 0

            overhead_f = ((parallel_f + layer_num_filters - 1) // parallel_f) * parallel_f - layer_num_filters
            overhead_f /= parallel_f

            overhead_h = ((parallel_h + layer_ofms_height - 1) // parallel_h) * parallel_h - layer_ofms_height
            overhead_h /= layer_ofms_height

            overhead_w = ((parallel_w + layer_ofms_width - 1) // parallel_w) * parallel_w - layer_ofms_width
            overhead_w /= layer_ofms_width
            
            layers_overheads.append(layers_num_of_ops[conv_layer_index] * \
                                    ( (1 + overhead_f) * (1 + overhead_h) * (1 + overhead_w) - 1))
            conv_layer_index += 1

    return layers_overheads

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


fms_sizes = get_fms_sizes(model_dag)
weight_sizes = get_weights_sizes(model_dag)
# for weigh_size in weight_sizes:
#     print(weigh_size)

print('weights:', sum(weight_sizes)/1000000)
print('weights up to:', sum(weight_sizes[:13])/1000000)

layers_num_of_ops = utils.get_layers_op_counts(model_dag)
dw_layers_num_of_ops = utils.get_dw_laye_op_counts(model_dag)

sum_ops = sum(layers_num_of_ops)
sesl_ops = sum(layers_num_of_ops[0:6])

sum_dw_ops = sum(dw_layers_num_of_ops)
print('GOPs:', sum_ops/1000000000)
print('dw / all:', sum_dw_ops / sum_ops)

sum_seml_ops = sum(layers_num_of_ops[6:])
sum_seml_dw_ops = sum(dw_layers_num_of_ops[2:])
print('seml DW GOPs:', sum_seml_dw_ops / 1000000000)
print('seml PW GOPs:', (sum_seml_ops - sum_seml_dw_ops) / 1000000000)
print('seml dw / seml pw:', sum_seml_dw_ops / (sum_seml_ops - sum_seml_dw_ops) )
print('dw / pw:', sum_dw_ops / (sum_seml_ops - sum_dw_ops) )

print('seml / all: ', sum_seml_ops / sum_ops)
print('sesl / all: ', sesl_ops / sum_ops)
print('sesl / seml: ', sesl_ops / sum_seml_ops)

print([layers_num_of_ops[i] / sum_ops for i in range(len(layers_num_of_ops))])
# for i in range(len(layers_num_of_ops)):
#     print(layers_num_of_ops[i] / (fms_sizes[i] + weight_sizes[i]))
# print ops
#sum_ops_so_far = 0

# print('***************************************')

# for layer_specs in model_dag:
#     if 'type' in layer_specs and layer_specs['type'] in ['s', 'dw', 'pw']:
#         print(layer_specs['ifms_shape'][1])

layers_overheads = get_layers_overheads(model_dag, layers_num_of_ops, 8, 8, 8)

print([layers_overheads[i] / layers_num_of_ops[i] for i in range(len(layers_num_of_ops))])
print('all overhead:', sum(layers_overheads) / sum_ops)