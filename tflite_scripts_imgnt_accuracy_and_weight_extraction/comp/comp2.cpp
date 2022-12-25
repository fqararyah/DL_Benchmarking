expansion_block = "//****************************\n \
const int layer_*i*_pw_num_fils = *LNF* / alpha;\n \
const int layer_*i*_pw_depth = layer_*i-1_pw*_num_fils;\n \
const int layer_*i*_pw_ifm_height = layer_*i-1_pw*_ofm_height;\n \
const int layer_*i*_pw_ifm_width = layer_*i-1_pw*_ofm_width;\n \
const int layer_*i*_pw_ofm_height = layer_*i*_pw_ifm_height;\n \
const int layer_*i*_pw_ofm_width = layer_*i*_pw_ifm_width;\n \
const int layer_*i*_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_*i*_pw_depth / pw_tile_d);\n \
const int layer_*i*_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_*i*_pw_num_fils / pw_conv_parallelism_out);\n \
const int layer_*i*_pw_num_of_tiles_w = (int)(0.99 + (float)layer_*i*_pw_ofm_width / pw_tile_w); \n \
const int layer_*i*_pw_num_of_tiles_h = (int)(0.99 + (float)layer_*i*_pw_ofm_height / pw_tile_h); \n \
const int layer_*i*_pw_num_of_weight_groups_for_one_pass = layer_*i*_pw_depth * pw_conv_parallelism_out / weights_group_items; \n \
const int layer_*i*_pw_weights_offset = *LWOF*; \n \
const int layer_*i*_relu = 6;\n\
//****************************\n"