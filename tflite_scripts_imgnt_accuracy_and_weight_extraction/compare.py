from xml import dom
from xml.dom.minidom import DOMImplementation
import numpy as np

from models_archs import utils

to_compare_layer_index = 7

import sys

if(len(sys.argv) > 1):
    to_compare_layer_index = int(sys.argv[1])

layers_ofms_shape = utils.read_layers_output_shapes()
skip_connections_indices = utils.read_skip_connections_indices()

tf_lite_to_my_cnn_layer_mapping = {0:2}
skip_connections_so_far = 0
for layer_index in range(1, len(layers_ofms_shape)):
    if layer_index + 1 in skip_connections_indices:
        skip_connections_so_far += 1
    tf_lite_to_my_cnn_layer_mapping[layer_index] = layer_index + 1 + skip_connections_so_far

#print(tf_lite_to_my_cnn_layer_mapping)
#layers_ofms_shape = {0: (32, 112, 112), 3: (16, 112, 112), 6: (24, 56, 56), 4: (96, 112, 112), 5: (96, 56, 56)}
print(to_compare_layer_index, tf_lite_to_my_cnn_layer_mapping[to_compare_layer_index])
domain_file = './scratch_out/ofms_{}.txt'.format(to_compare_layer_index)
range_file = './fms/fms_{}_{}_{}_{}.txt'.format(tf_lite_to_my_cnn_layer_mapping[to_compare_layer_index],\
    layers_ofms_shape[to_compare_layer_index].depth, layers_ofms_shape[to_compare_layer_index].height,\
    layers_ofms_shape[to_compare_layer_index].width)

ofms_hw = layers_ofms_shape[to_compare_layer_index].height * layers_ofms_shape[to_compare_layer_index].width
ofms_w = layers_ofms_shape[to_compare_layer_index].width
domain = np.loadtxt(domain_file).astype(np.int8)
rng = np.loadtxt(range_file).astype(np.int8)

# sum = 0
# for i in range(rng.size):
#     if (int(domain[i]) - rng[i] > 1 or int(domain[i]) - rng[i] < -1 ) and rng[i] != -128 and domain[i] != -128:
#         sum += int(domain[i]) - rng[i]

# print(sum)
# print(sum/rng.size)

sum = 0
cnt1=0
cnt2=0
cnt3=0
diff_map = {}
diff_locs = {}
for i in range(rng.size):
    if int(domain[i]) - rng[i] != 0:
        sum += abs(int(domain[i]) - rng[i])
        if domain[i] not in diff_map:
            diff_map[domain[i]] = {}
        if rng[i] not in diff_map[domain[i]]:    
            diff_map[domain[i]][rng[i]] = 0
        diff_map[domain[i]][rng[i]] += 1
    
    if domain[i] == rng[i]:
        cnt1 += 1
    elif int(domain[i]) - rng[i] !=0:
        cnt2 +=1
        d = int(i / ofms_hw)
        h = int((i % ofms_hw) / ofms_w)
        w = int(i % ofms_w) 
        position = (d, h, w)
        if int(domain[i]) - rng[i] > 2 or int(domain[i]) - rng[i] < -2:
            diff_locs[position] = (domain[i], rng[i])
            #print(domain[i], rng[i])
            cnt3 += 1
        #    print(domain[i], rng[i])

# for key, val in diff_map.items():
#     print(key, val)
#     print('***************')

count = 0
for key, val in diff_locs.items():
    print(key, val)
    print('***************')
    count += 1
    if count > 100:
        break

print(sum)
print(sum/rng.size)
print('equal: ', cnt1)
print('different: ', cnt2)
print('very different: ', cnt3)