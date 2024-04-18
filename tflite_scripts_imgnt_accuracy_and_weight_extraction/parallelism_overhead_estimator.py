from fcntl import F_SETFL
from models_archs import utils
import math

MODEL_NAME = 'mob_v2'  # uniform_mobilenetv2_75

utils.set_globals(MODEL_NAME, MODEL_NAME)

def get_layers_theoritical_overheads(model_dag, layers_num_of_ops, parallel_f, parallel_d, parallel_h, parallel_w):    

    conv_layer_index = 0
    layers_overheads = []
    for layer_specs in model_dag:
        if 'type' in layer_specs and layer_specs['type'] in ['s', 'pw', 'dw']:
            filter_dim = 1
            layers_ofms_shape = layer_specs['ofms_shape']
            layers_ifms_shape = layer_specs['ifms_shape']
            layers_weights_shape = layer_specs['weights_shape']
            if layer_specs['type'] in ['s', 'dw']:
                filter_dim = layers_weights_shape[-1]
            layer_num_filters = layers_weights_shape[0]
            layer_ifms_depth = layers_ifms_shape[0] 
            layer_ofms_height = layers_ofms_shape[1] 
            layer_ofms_width = layers_ofms_shape[2]
            
            ideal_ops = layers_num_of_ops[conv_layer_index]
            actual_f = parallel_f * ((layer_num_filters + parallel_f - 1) // parallel_f)
            actual_d = parallel_d * ((layer_ifms_depth + parallel_d - 1) // parallel_d)
            actual_h = parallel_h * ((layer_ofms_height + parallel_h - 1) // parallel_h)
            actual_w = parallel_w * ((layer_ofms_width + parallel_w - 1) // parallel_w)

            if layer_specs['type'] in ['dw']:
                actual_ops = 2 * actual_d * actual_h * actual_w * filter_dim * filter_dim
            else:
                actual_ops = 2 * actual_f * actual_d * actual_h * actual_w * filter_dim * filter_dim

            layers_overheads.append(actual_ops - ideal_ops)
            #print(actual_f, actual_d, actual_h, actual_w, filter_dim)
            #print(actual_ops, ideal_ops, (actual_ops - ideal_ops) / ideal_ops)

            conv_layer_index += 1

    return layers_overheads

model_dag = model_dag = utils.read_model_dag()

layers_num_of_ops = utils.get_layers_op_counts(model_dag)

layers_overheads = get_layers_theoritical_overheads(model_dag, layers_num_of_ops, 8, 1, 4, 8)
print((sum(layers_overheads) + sum(layers_num_of_ops)) / sum(layers_num_of_ops))
layers_overheads = get_layers_theoritical_overheads(model_dag, layers_num_of_ops, 16, 1, 2, 8)
print((sum(layers_overheads) + sum(layers_num_of_ops)) / sum(layers_num_of_ops))
layers_overheads = get_layers_theoritical_overheads(model_dag, layers_num_of_ops, 32, 1, 1, 8)
print((sum(layers_overheads) + sum(layers_num_of_ops)) / sum(layers_num_of_ops))