from fcntl import F_SETFL
from models_archs import utils
import analysis_utils

MODEL_NAME = 'prox'

utils.set_globals(MODEL_NAME, MODEL_NAME)

layers_types = utils.read_layers_types()
layers_weights = utils.read_layers_weight_shapes(layers_types)
layers_inputs = utils.read_layers_input_shapes()
layers_outputs = utils.read_layers_output_shapes()
layers_strides = utils.read_layers_strides()

weight_reuse = []
activation_reuse = []
weights_sizes = []
activations_sizes = []
weight_reuse_and_ops = {}
activation_reuse_and_ops = {}
layers_num_of_ops = []

activation_resuse_thresholds = {9: 0, 16: 0, 24: 0, 25: 0, 32: 0, 40: 0, 48: 0, 49: 0, 64: 0, 72: 0, 80: 0, 96: 0, 120: 0, 128: 0,
                                144: 0, 160: 0, 192: 0, 240: 0, 256: 0, 288: 0, 320: 0, 348: 0, 480: 0, 512: 0, 576: 0, 960: 0, 1000: 0, 1152: 0, 1280: 0}


def print_dw_ops():
    layers_num_of_ops = analysis_utils.get_layers_num_of_ops(
        layers_inputs, layers_weights, layers_types, layers_strides)
    for i in range(len(layers_types)):
        if layers_types[i] == 'dw':
            print(layers_num_of_ops[i])


def print_weights_sizes(print_it=False):
    for weights_shape in layers_weights:
        weights_sizes.append(weights_shape.depth * weights_shape.height *
                             weights_shape.width * weights_shape.num_of_filters)
        if print_it:
            print(weights_sizes[-1])


def print_fms_reuse(print_it=False):
    i = 0
    for weights_shape in layers_weights:
        if layers_types[i] == 'pw':
            if print_it:
                print(weights_shape.num_of_filters)
            activation_reuse.append(weights_shape.num_of_filters)
        elif layers_types[i] == 'dw':
            if print_it:
                print(weights_shape.height * weights_shape.width)
            activation_reuse.append(weights_shape.height * weights_shape.width)
        elif layers_types[i] == 's':
            if print_it:
                print(weights_shape.height * weights_shape.width *
                      weights_shape.num_of_filters)
            activation_reuse.append(
                weights_shape.height * weights_shape.width * weights_shape.num_of_filters)
        i += 1


def print_fms_sizes(print_it=False):
    sum = 0
    for i in range(len(layers_inputs)):
        activations_sizes.append(layers_inputs[i].depth * layers_inputs[i].height *
        layers_inputs[i].width + layers_outputs[i].depth * layers_outputs[i].height *
        layers_outputs[i].width)
        if print_it:
            print(activations_sizes[-1])

        sum += layers_inputs[i].depth * layers_inputs[i].height * \
            layers_inputs[i].width + layers_outputs[i].depth * layers_outputs[i].height * \
            layers_outputs[i].width
    print('all(in millions):', sum/1000000)


def print_weight_reuse(print_it=False):
    for i in range(len(layers_inputs)):
        if print_it:
            print(layers_outputs[i].height * layers_outputs[i].width)
        weight_reuse.append(layers_outputs[i].height * layers_outputs[i].width)


def reuse_and_ops():
    print_weight_reuse()
    print_fms_reuse()

    layers_num_of_ops = analysis_utils.get_layers_num_of_ops(
        layers_inputs, layers_weights, layers_types, layers_strides)

    totla_ops = sum(layers_num_of_ops)

    for i in range(len(weight_reuse)):
        if weight_reuse[i] not in weight_reuse_and_ops:
            weight_reuse_and_ops[weight_reuse[i]] = 0
        if activation_reuse[i] not in activation_reuse_and_ops:
            activation_reuse_and_ops[activation_reuse[i]] = 0

        weight_reuse_and_ops[weight_reuse[i]] += layers_num_of_ops[i]
        activation_reuse_and_ops[activation_reuse[i]] += layers_num_of_ops[i]

    w_reuse_keys = weight_reuse_and_ops.keys()
    w_reuse_vals = weight_reuse_and_ops.values()

    zipped_lists = zip(w_reuse_keys, w_reuse_vals)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    w_reuse_keys, w_reuse_vals = [list(tuple) for tuple in tuples]

    for threshold in activation_resuse_thresholds.keys():
        for reuse, ops in activation_reuse_and_ops.items():
            if reuse >= threshold:
                activation_resuse_thresholds[threshold] += ops

    for i in range(len(w_reuse_keys) - 2, -1, -1):
        w_reuse_vals[i] += w_reuse_vals[i+1]

    # for i in range(len(a_reuse_keys)):
    #     print(a_reuse_keys[i])

    # print('*****')

    for thresold, ops in activation_resuse_thresholds.items():
        print(ops/totla_ops)

    print('*****')

    for i in range(len(w_reuse_keys)):
        print(w_reuse_keys[i])

    print('*****')

    for i in range(len(w_reuse_keys)):
        print(w_reuse_vals[i]/totla_ops)


#reuse_and_ops()

print_fms_reuse()
print_fms_sizes()
print_weight_reuse()
print_weights_sizes()

layers_num_of_ops = analysis_utils.get_layers_num_of_ops(
        layers_inputs, layers_weights, layers_types, layers_strides)

avg_weight_reuse = 0
avg_act_reuse = 0
for i in range(len(weight_reuse)):
    avg_weight_reuse += weight_reuse[i] * weights_sizes[i]

for i in range(len(activation_reuse)):
    avg_act_reuse += activation_reuse[i] * activations_sizes[i]

print(avg_weight_reuse/ sum(weights_sizes))
print(avg_act_reuse / sum(activations_sizes))
