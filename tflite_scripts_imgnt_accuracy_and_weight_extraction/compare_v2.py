from xml import dom
from xml.dom.minidom import DOMImplementation
import numpy as np
import sys
from models_archs import utils

utils.NET_PREFIX = 'mob_v2'
utils.set_globals(utils.NET_PREFIX, utils.NET_PREFIX)

to_compare_layer_index = 7

ref = ''
if(len(sys.argv) > 1):
    to_compare_layer_index = int(sys.argv[1])
if(len(sys.argv) > 2):
    ref = sys.argv[2]

layers_ofms_shape = utils.read_layers_output_shapes()
skip_connections_indices = utils.read_skip_connections_indices()
layers_execution_sequence = utils.read_layers_execution_sequence()

domain_file = './scratch_out/ofms_{}'.format(to_compare_layer_index)
if len(ref)>0:
    domain_file += '_' + ref + '.txt'
else:
    domain_file += '.txt'

#domain_file = './scratch_out/ofms_1.txt'

#range_file = './scratch_out/ofms_{}_ref.txt'.format(to_compare_layer_index)
to_compare_fms = str(to_compare_layer_index + 1)
conv_count = 0
layer_index = 0
non_standard = False
while conv_count < (to_compare_layer_index + 1) + 1 and to_compare_layer_index != len(layers_ofms_shape) - 1:
    non_standard = True
    if layers_execution_sequence[layer_index] == 'conv2d':
        conv_count += 1
    layer_index += 1

print(layer_index, layers_execution_sequence[layer_index - 1])
if non_standard and layers_execution_sequence[layer_index - 2] != 'conv2d' and 'pad' not in layers_execution_sequence[layer_index - 2]:
    to_compare_fms += '_' + layers_execution_sequence[layer_index - 2]



range_file = './{}/fms/fms_conv2d_{}_{}_{}_{}.txt'.format(utils.NET_PREFIX ,to_compare_fms,\
     layers_ofms_shape[to_compare_layer_index].depth, layers_ofms_shape[to_compare_layer_index].height,\
     layers_ofms_shape[to_compare_layer_index].width)

#range_file = './eff_b0/fms/fms_conv2d_2_mul_1_2_32_112_112.txt'
print(range_file)

ofms_hw = layers_ofms_shape[to_compare_layer_index].height * layers_ofms_shape[to_compare_layer_index].width
ofms_w = layers_ofms_shape[to_compare_layer_index].width
domain = np.loadtxt(domain_file).astype(np.int8)
rng = np.loadtxt(range_file).astype(np.int8)

# sum = 0
# for i in range(rng.size):
#     if (int(domain[i]) - rng[i] > 1 or int(domain[i]) - rng[i] < -1 ) and rng[i] != -128 and domain[i] != -128:
#         sum += int(domain[i]) - rng[i]

# print(sum)
print('(calculated, expected)')

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
        if int(domain[i]) - rng[i] < -1 or int(domain[i]) - rng[i] > 1:
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

print('max= ', np.max( np.abs(domain - rng)))
print(sum)
print(sum/rng.size)
print('equal: ', cnt1)
print('different: ', cnt2)
print('very different: ', cnt3)