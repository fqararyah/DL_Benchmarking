import sys
import pathlib


current_dir = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(current_dir)
print(current_dir)
import classes

DELIMITER = '::'

NET_PREFIX = 'mob_v2'
NET_FULL_NAME = 'mobilenet_v2'
input_folder = current_dir + '/models/'\
     + NET_FULL_NAME + '/'
IFMS_FILE = input_folder + 'layers_inputs.txt'
OFMS_FILE = input_folder + 'layers_outputs.txt'
LAYERS_TYPES_FILE = input_folder + 'layers_types.txt'
LAYERS_WEIGHTS_FILE = input_folder + 'layers_weights.txt'
LAYERS_STRIDES_FILE = input_folder + 'layers_strides.txt'
EXPANSION_PROJECTION_FILE = input_folder + 'expansion_projection.txt'
LAYERS_RELUS_FILE = input_folder + 'layers_relus.txt'

def set_globals(prefix, full_name):
    global NET_PREFIX, NET_FULL_NAME, input_folder, IFMS_FILE, OFMS_FILE, LAYERS_TYPES_FILE, LAYERS_WEIGHTS_FILE, LAYERS_STRIDES_FILE\
        ,EXPANSION_PROJECTION_FILE, LAYERS_RELUS_FILE
    NET_PREFIX = prefix
    NET_FULL_NAME = full_name
    input_folder = './models/' + NET_FULL_NAME + '/'
    IFMS_FILE = input_folder + 'layers_inputs.txt'
    OFMS_FILE = input_folder + 'layers_outputs.txt'
    LAYERS_TYPES_FILE = input_folder + 'layers_types.txt'
    LAYERS_WEIGHTS_FILE = input_folder + 'layers_weights.txt'
    LAYERS_STRIDES_FILE = input_folder + 'layers_strides.txt'
    EXPANSION_PROJECTION_FILE = input_folder + 'expansion_projection.txt'
    LAYERS_RELUS_FILE = input_folder + 'layers_relus.txt'


def clean_line(line):
    return line.replace(' ', '').replace('\n', '')


def read_layers_input_shapes():
    layers_inputs = []
    with open(IFMS_FILE, 'r') as f:
       for line in f:
            line = clean_line(line)
            splits = line.split('x')
            if len(splits) > 0:
                layers_inputs.append(classes.feature_map(
                    int(splits[0]), int(splits[1]), int(splits[2]) ))

    return layers_inputs


def read_layers_output_shapes():
    layers_outputs = []
    with open(OFMS_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            splits = line.split('x')
            if len(splits) > 0:
                splits = line.split('x')
                layers_outputs.append(classes.feature_map(
                    int(splits[0]), int(splits[1]), int(splits[2]) ))

    return layers_outputs


def read_layers_weight_shapes(layers_types):
    layers_weights = []
    count = 0
    with open(LAYERS_WEIGHTS_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            splits = line.split('x')
            if len(splits) > 0:
                if layers_types[count] == 'dw':
                   layers_weights.append(classes.weights(int(splits[0]), 1,
                                                int(splits[1]) if len(splits) > 1 else 1, int(splits[2]) if len(splits) > 2 else 1)) 
                else:
                    layers_weights.append(classes.weights(int(splits[0]), int(splits[1]),
                                                int(splits[2]) if len(splits) > 2 else 1, int(splits[3]) if len(splits) > 3 else 1))
                count += 1

    return layers_weights


def read_layers_strides():
    layers_strides = []
    with open(LAYERS_STRIDES_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            line = clean_line(line)
            layers_strides.append(int(line))

    return layers_strides

def read_layers_types():
    layers_types = []
    with open(LAYERS_TYPES_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            line = clean_line(line)
            layers_types.append(line)

    return layers_types

def read_expansion_projection():
    expansion_projection = []
    with open(EXPANSION_PROJECTION_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            line = clean_line(line)
            expansion_projection.append(int(line))

    return expansion_projection

def read_layers_relus():
    layers_relus = []
    with open(LAYERS_RELUS_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            line = clean_line(line)
            layers_relus.append(int(line))

    return layers_relus