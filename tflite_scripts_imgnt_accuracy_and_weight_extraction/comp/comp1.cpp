void fill_weights_tile_from_weight_groups_tile(
		weights_grp_dt weight_groups_buffer[num_of_weight_groups_in_the_largest_weight_tile],
		weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
		int starting_filter, const int layer_depth,
		const int num_of_weight_groups, const int layer_weights_offset) {
#pragma HLS INLINE off

//assumes pw_parallelism_out * filter depth is divisable by weight group number

	fill_weights_loop: for (int weight_grp_index = 0;
			weight_grp_index < num_of_weight_groups; weight_grp_index++) {
		weights_grp_dt chunck = weight_groups_buffer[weight_grp_index];
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