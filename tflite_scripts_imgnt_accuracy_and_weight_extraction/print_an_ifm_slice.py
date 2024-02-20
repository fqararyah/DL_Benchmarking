from xml import dom
from xml.dom.minidom import DOMImplementation
import numpy as np

from models_archs import utils
import sys

model = 'resnet50'
utils.set_globals(model, model)

to_print_layer_index = 3
slice_index = 0
slice_direction = 0
directions_map = {0:'hw', 1:'hd'}


if(len(sys.argv) > 1):
    to_print_layer_index = int(sys.argv[1])
if(len(sys.argv) > 2):
    slice_index = int(sys.argv[2])
if(len(sys.argv) > 3):
    slice_direction = int(sys.argv[3])


model_dag = utils.read_model_dag()
layer_ifms_shape = model_dag[to_print_layer_index]['ifms_shape']

ifms_file = './{}/fms/ifms_{}.txt'.format(utils.NET_PREFIX, to_print_layer_index)

slice_file = './scratch_out/ifms_{}_slice_{}_{}.txt'.format(to_print_layer_index, slice_index, directions_map[slice_direction])

arr = np.loadtxt(ifms_file).astype(np.int8)

arr = np.reshape(arr, (layer_ifms_shape[0], layer_ifms_shape[1],layer_ifms_shape[2]))

if directions_map[slice_direction] == 'hw':
    to_print = arr[slice_index,:,:]
elif directions_map[slice_direction] == 'hd':
    to_print = arr[:,:,slice_index]
    to_print = np.transpose(to_print, (1,0))

np.savetxt(slice_file, to_print, fmt='%i')