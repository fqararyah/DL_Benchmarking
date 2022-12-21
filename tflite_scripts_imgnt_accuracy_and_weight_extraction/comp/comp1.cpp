fill_dw_layer_weights(dw_weights_50, dw_weights_buffer, layer_50_dw_depth, layer_50_dw_filter_size, layer_50_dw_filter_size);
dw_conv_3x3(dw_weights_buffer, result2, channels, 50, layer_50_dw_depth,
			layer_50_dw_ifm_width, layer_50_dw_ifm_height, layer_50_dw_num_of_tiles_in_d,
			layer_50_dw_num_of_tiles_h, layer_50_dw_num_of_tiles_w,
			layer_50_dw_strides, layer_50_dw_padding_left, layer_50_dw_padding_right, layer_50_dw_padding_top,
			1, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result2, 51, layer_51_pw_depth,
		layer_51_pw_num_fils, layer_51_pw_num_of_tiles_in_d,
		layer_51_pw_num_of_tiles_out_d, layer_51_pw_num_of_tiles_h,
		layer_51_pw_num_of_tiles_w, tmp_channels, 0,
		layer_51_pw_num_of_weight_groups_for_one_pass,
		0, layer_51_pw_weights_offset, layer_51_relu, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result2, 52, layer_52_pw_depth,
		layer_52_pw_num_fils, layer_52_pw_num_of_tiles_in_d,
		layer_52_pw_num_of_tiles_out_d, layer_52_pw_num_of_tiles_h,
		layer_52_pw_num_of_tiles_w, tmp_channels, 0,
		layer_52_pw_num_of_weight_groups_for_one_pass,
		1, layer_52_pw_weights_offset, layer_52_relu, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);