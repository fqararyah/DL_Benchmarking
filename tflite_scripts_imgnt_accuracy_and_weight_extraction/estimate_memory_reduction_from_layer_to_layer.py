from models_archs import utils

utils.NET_PREFIX = 'resnet101'
utils.set_globals(utils.NET_PREFIX, utils.NET_PREFIX)

model_dag = model_dag = utils.read_model_dag()

def get_max_fms_buffer(model_dag, starting_layer):

    max_buffer = 0
    starting_layer_act_index = 0
    conv_layers_so_far = 0
    while conv_layers_so_far < starting_layer:
        starting_layer_act_index += 1
        layer_specs = model_dag[starting_layer_act_index]
        if 'type' in layer_specs and layer_specs['type'] in ['s', 'dw', 'pw']:
            conv_layers_so_far += 1

    for layer_index in range(starting_layer_act_index, len(model_dag)):
        layer_specs = model_dag[layer_index]
        if 'type' in layer_specs and layer_specs['type'] in ['s', 'dw', 'pw']:
            input_shape = layer_specs['ifms_shape']
            input_size = input_shape[0] * input_shape[1] * input_shape[2]
            output_shape = layer_specs['ofms_shape']
            output_size = input_shape[0] * input_shape[1] * input_shape[2]

            if len(layer_specs['children']) > 1:
                output_size *= 2

            memory_cons = input_size + output_size
            if memory_cons > max_buffer:
                max_buffer = memory_cons

    return max_buffer

full_model_max_buffer = get_max_fms_buffer(model_dag, 0)
print(len(model_dag), full_model_max_buffer)
for i in range(1, 16, 2):
    current_max_buffer = get_max_fms_buffer(model_dag, i)
    print(current_max_buffer / full_model_max_buffer)


    