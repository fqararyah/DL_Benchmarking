from cProfile import label
from cmath import nan
from ctypes import util

from click import style
from numpy import append, pad
from torch import avg_pool1d, fmax
from models_archs import utils
import matplotlib as mpl
import analysis_utils
import matplotlib.pyplot as plt
import math

utils.set_globals('prox', 'prox')

layers_types = utils.read_layers_types()
layers_weights = utils.read_layers_weight_shapes(layers_types)
layers_inputs = utils.read_layers_input_shapes()
layers_outputs = utils.read_layers_output_shapes()
layers_strides = utils.read_layers_strides()

layers_num_of_ops = analysis_utils.get_layers_num_of_ops(
            layers_inputs, layers_weights, layers_types, layers_strides)

sum_pw = 0
sum_pw_weights = 0
sum_dw = 0
sum_conv = 0
sum_so_far = 0
avg_depth = 0
sum_3 = 0
sum_5 = 0
sum_7 = 0
sum_lt_14 = 0

for i in range(9, len(layers_num_of_ops)):
    avg_depth += layers_inputs[i].depth
    sum_so_far += layers_num_of_ops[i]
    #print(i, sum_so_far / sum(layers_num_of_ops))
    if layers_types[i] == 'pw':
        sum_pw += layers_num_of_ops[i]
        sum_pw_weights += layers_weights[i].get_size()
    elif layers_types[i] == 'dw':
        print(layers_num_of_ops[i], layers_inputs[i].width)
        sum_dw += layers_num_of_ops[i]
        if layers_weights[i].height == 5:
            sum_5 += layers_num_of_ops[i]
        if layers_weights[i].width == 3:
            sum_3 += layers_num_of_ops[i]
        if layers_weights[i].width == 7:
            sum_7 += layers_num_of_ops[i]
        if layers_inputs[i].width <= 14:
            sum_lt_14 += layers_num_of_ops[i]

    elif layers_types[i] == 'c':
        sum_conv += layers_num_of_ops[i]

PEs = 1024
non_pw_PEs = 128
avg_depth /= len(layers_inputs)
print('avg depth:', avg_depth)

print('sum of pw:', sum_pw)
print('sum of pw weights:', sum_pw_weights)
print('sum of dw:', sum_dw)
print('ratio of dw:', sum_dw / (sum_conv + sum_dw + sum_pw))
print('ratio of dw and conv:', (sum_dw + sum_conv) / (sum_conv + sum_dw + sum_pw))
#print('ratio of dw to conv:', sum_dw/ sum_conv)
print('sum of all ops:', sum_conv + sum_dw + sum_pw)

print('ratio of pw:', sum_pw / (sum_conv + sum_dw + sum_pw), PEs * sum_pw / (sum_conv + sum_dw + sum_pw))
print('ratio of conv:', sum_conv / (sum_conv + sum_dw), non_pw_PEs * sum_conv / (sum_conv + sum_dw))
print('sum_3 ratio:', (sum_3) / (sum_3 + sum_5 + sum_7), non_pw_PEs * (sum_3) / (sum_conv + sum_dw) )
print('sum_5 ratio:', (sum_5) / (sum_3 + sum_5 + sum_7), non_pw_PEs * (sum_5) / (sum_conv + sum_dw) )
print('sum_7 ratio:', (sum_7) / (sum_3 + sum_5 + sum_7), non_pw_PEs * (sum_7) / (sum_conv + sum_dw) )
print('ratio_gt_14 ratio:', (sum_lt_14) / ( sum_dw))