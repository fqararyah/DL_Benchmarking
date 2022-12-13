void fill_weights_tile_off_chip(weights_grp_dt *weights,
		weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
		int starting_filter, const int layer_depth,
		const int num_of_weight_groups, const int layer_weights_offset) {
//assumes pw_parallelism_out * filter depth is divisable by weight group number
	const int current_fill_offset = layer_weights_offset
			+ starting_filter * layer_depth / weights_group_items;

	fill_weights_loop: for (int weight_grp_index = 0;
			weight_grp_index < num_of_weight_groups; weight_grp_index++) {
		weights_grp_dt chunck = weights[current_fill_offset + weight_grp_index];
		for (int within_filter_index = 0;
				within_filter_index
						< num_of_weights_in_the_same_filter_and_group;
				within_filter_index++) {
#pragma HLS UNROLL
			for (int filter_index = 0; filter_index < pw_conv_parallelism_out;
					filter_index++) {
#pragma HLS UNROLL
				weights_tile[filter_index][weight_grp_index
						* num_of_weights_in_the_same_filter_and_group
						+ within_filter_index] = (weights_dt) chunck(
						(within_filter_index * pw_conv_parallelism_out
								+ filter_index) * weights_dt_width
								+ weights_dt_offset,
						(within_filter_index * pw_conv_parallelism_out
								+ filter_index) * weights_dt_width);
			}
		}
	}
}