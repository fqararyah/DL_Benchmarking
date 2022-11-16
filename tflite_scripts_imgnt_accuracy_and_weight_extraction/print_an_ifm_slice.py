from xml import dom
from xml.dom.minidom import DOMImplementation
import numpy as np

from models_archs import utils

to_print_layer_index = 7
slice_index = 0

import sys

if(len(sys.argv) > 1):
    to_print_layer_index = int(sys.argv[1])
if(len(sys.argv) > 2):
    slice_index = int(sys.argv[2])

layers_ofms_shape = utils.read_layers_output_shapes()
skip_connections_indices = utils.read_skip_connections_indices()

tf_lite_to_my_cnn_layer_mapping = {0:2}
skip_connections_so_far = 0
for layer_index in range(1, len(layers_ofms_shape)):
    if layer_index + 1 in skip_connections_indices:
        skip_connections_so_far += 1
    tf_lite_to_my_cnn_layer_mapping[layer_index] = layer_index + 1 + skip_connections_so_far

ifms_file = './fms/fms_{}_{}_{}_{}.txt'.format(tf_lite_to_my_cnn_layer_mapping[to_print_layer_index],\
     layers_ofms_shape[to_print_layer_index].depth, layers_ofms_shape[to_print_layer_index].height,\
     layers_ofms_shape[to_print_layer_index].width)

slice_file = './scratch_out/layer_{}_slice_{}.txt'.format(to_print_layer_index, slice_index)

arr = np.loadtxt(ifms_file).astype(np.int8)

arr = np.reshape(arr, (layers_ofms_shape[to_print_layer_index].depth, layers_ofms_shape[to_print_layer_index].height,\
     layers_ofms_shape[to_print_layer_index].width))

to_print = arr[slice_index,:,:]
np.savetxt(slice_file, to_print, fmt='%i')

# for i in range(layers_ofms_shape[to_print_layer_index].height):
#     line = ''
#     for j in range(layers_ofms_shape[to_print_layer_index].width):
#         line += str(to_print[i][j])
#     print(line)