from cgi import FieldStorage
from cmath import sqrt
#import utils
import gc
from math import ceil, floor
import math


def accumulated_weights(weights):
    accumulated = []
    for i in range(len(weights)):
        if i == 0:
            accumulated.append(weights[i].get_size())
        else:
            accumulated.append(accumulated[i - 1] + weights[i].get_size())

    return accumulated


def get_layers_num_of_ops(layers_inputs, layers_weights, layers_types, layers_strides):
    layers_ops = []
    for i in range(len(layers_weights)):
        if layers_types[i] == 'dw':
            layers_ops.append(int(layers_inputs[i].depth * layers_inputs[i].height *
                                  layers_inputs[i].width *
                                  layers_weights[i].height * layers_weights[i].width / (layers_strides[i]**2)))
        else:
            layers_ops.append(int(layers_inputs[i].depth * layers_inputs[i].height *
                                  layers_inputs[i].width * layers_weights[i].num_of_filters *
                                  layers_weights[i].height * layers_weights[i].width / (layers_strides[i]**2)))

    return layers_ops


def get_layer_types_rations(layers_types, layers_ops):
    sum_pw = 0
    sum_dw = 0
    sum_conv = 0
    for i in range(len(layers_types)):
        if layers_types[i] == 'pw':
            sum_pw += layers_ops[i]
        elif layers_types[i] == 'dw':
            sum_dw += layers_ops[i]
        elif layers_types[i] == 'c':
            sum_conv += layers_ops[i]
    return [sum_pw, sum_dw, sum_conv]


def get_ideal_time(layers_types, layers_ops, num_of_pes, first_conv_in_pipeline=True):
    if 'pw' in layers_types:
        [sum_pw, sum_dw, sum_conv] = get_layer_types_rations(
            layers_types, layers_ops)
        pw_pes = num_of_pes * sum_pw / (sum_pw + sum_dw)
        dw_pes = num_of_pes * sum_dw / (sum_pw + sum_dw)
        if first_conv_in_pipeline:
            return sum_dw / dw_pes + sum_pw / pw_pes + sum_conv / pw_pes
        else:
            return sum_dw / dw_pes + sum_pw / pw_pes

    return sum(layers_ops) / num_of_pes


def uk_overhead(num_of_pes, split_point, layers_inputs, layers_weights, layers_types, layers_strides, pure_uk, print_pes=False):

    layers_ops = get_layers_num_of_ops(
        layers_inputs, layers_weights, layers_types, layers_strides)
    original_num_of_pes = num_of_pes
    dw_pes = 0

    if 'pw' in layers_types:
        [sum_pw, sum_dw, sum_conv] = get_layer_types_rations(
            layers_types, layers_ops)
        if pure_uk:
            pw_ratio = (sum_pw) / (sum_conv + sum_dw + sum_pw)
        else:
            pw_ratio = (sum_pw) / (sum_dw + sum_pw)
        num_of_pes = int(pw_ratio * num_of_pes)
    else:
        conv_ratio = (sum(layers_ops) - layers_ops[0]) / sum(layers_ops)
        num_of_pes = int(conv_ratio * num_of_pes)

    if 'dw' in layers_types:
        [sum_pw, sum_dw, sum_conv] = get_layer_types_rations(
            layers_types, layers_ops)
        dw_pes = ceil((sum_dw) / (sum_conv + sum_dw + sum_pw)
                       * original_num_of_pes)

    parallelism_on_a_dims = [1] * 4  # [filters, w, h, d]

    max_common_parallelism = [
        min(
            [layers_weights[i].num_of_filters for i in range(len(layers_inputs))]),
        min(
            [layers_inputs[i].width for i in range(len(layers_inputs))]),
        min(
            [layers_inputs[i].height for i in range(len(layers_inputs))]),
        min(
            [layers_inputs[i].depth for i in range(1, len(layers_inputs))])]

    saturated_dims = [False] * 4
    while True:
        candidate_dim_indx = -1
        for i in range(len(parallelism_on_a_dims)):
            if (candidate_dim_indx == -1 or parallelism_on_a_dims[i] < parallelism_on_a_dims[candidate_dim_indx]) and not saturated_dims[i]:
                candidate_dim_indx = i

        if candidate_dim_indx == -1:
            break

        desired_parallelism_on_candidate = parallelism_on_a_dims[candidate_dim_indx]
        for i in range(desired_parallelism_on_candidate + 1, max_common_parallelism[candidate_dim_indx] + 1):
            if max_common_parallelism[candidate_dim_indx] % i == 0:
                desired_parallelism_on_candidate = i
                break

        total_pes = 1
        for i in range(0, len(parallelism_on_a_dims)):
            if i != candidate_dim_indx:
                total_pes *= parallelism_on_a_dims[i]
            else:
                total_pes *= desired_parallelism_on_candidate

        if total_pes <= num_of_pes:
            parallelism_on_a_dims[candidate_dim_indx] = desired_parallelism_on_candidate
            if desired_parallelism_on_candidate == max_common_parallelism[candidate_dim_indx]:
                saturated_dims[candidate_dim_indx] = True
        else:
            saturated_dims[candidate_dim_indx] = True

    overheads = [0.0] * 3
    min_common_dims = [
        min(
            [layers_inputs[i].width for i in range(1, len(layers_inputs))]),
        min(
            [layers_inputs[i].height for i in range(1, len(layers_inputs))]),
        min(
            [layers_inputs[i].depth for i in range(1, len(layers_inputs))])]

    # for i in range(split_point, len(layers_inputs)):
    #     overheads[0] += layers_ops[i] * utils.AVERAGE_PARALLELIZED_LOOPS_PIPELINE_LATENCY / \
    #         ceil(min_common_dims[0] / parallelism_on_a_dims[1])
    #     overheads[1] += layers_ops[i] * utils.AVERAGE_PARALLELIZED_LOOPS_PIPELINE_LATENCY / \
    #         ceil(min_common_dims[1] / parallelism_on_a_dims[2])
    #     overheads[2] += layers_ops[i] * utils.AVERAGE_PARALLELIZED_LOOPS_PIPELINE_LATENCY / \
    #         ceil(min_common_dims[2] / parallelism_on_a_dims[3])

    if print_pes:
        print(original_num_of_pes, parallelism_on_a_dims)
    
    saved_dw = 0
    # if split_point > 0:
    #     for i in range(split_point):
    #         if layers_types[i] == 'dw':
    #             saved_dw += layers_ops[i] / dw_pes

    return min(overheads) * \
        (original_num_of_pes / (parallelism_on_a_dims[0]*parallelism_on_a_dims[1]*parallelism_on_a_dims[2]*parallelism_on_a_dims[3])) \
        / sum(layers_ops)


def get_pipeline_pes(split_point, layers_inputs, layers_weights, layers_types, layers_strides):
    layers_pes = []
    layers_ops = get_layers_num_of_ops(
        layers_inputs, layers_weights, layers_types, layers_strides)
    ops_gcd = math.gcd(layers_ops[1], layers_ops[2])

    for i in range(0, split_point):
        ops_gcd = math.gcd(ops_gcd, layers_ops[i])

    for i in range(0, split_point):
        layers_pes.append(int(layers_ops[i] / ops_gcd))

    return layers_pes


def get_memory_access_reduction(split_point, layers_inputs, layers_weights, sram_size):
    buffer_sizes = []
    for i in range(len(layers_inputs)):
        buffer_sizes.append(2 * layers_inputs[i].get_size())

    uk_space = sram_size - max(buffer_sizes)

    uk_weights = 0
    if sram_size - max(buffer_sizes) < sum([layers_weights[i].get_size() for i in range(len(layers_weights))]):
        for i in range(len(layers_weights)):
            if uk_weights + layers_weights[i].get_size() <= uk_space:
                uk_weights += layers_weights[i].get_size()

    hybrid_space = sram_size - max(buffer_sizes[split_point+1:])

    hybrid_weights = 0
    if sram_size - max(buffer_sizes) < sum([layers_weights[i].get_size() for i in range(len(layers_weights))]):
        for i in range(len(layers_weights)):
            if hybrid_weights + layers_weights[i].get_size() <= hybrid_space:
                hybrid_weights += layers_weights[i].get_size()

    return hybrid_weights - uk_weights
