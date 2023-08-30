#include "../headers/layers_imp_common_includes.h"
#include "../headers/dw_conv.h"
#include "../headers/pw_conv.h"

void fill_ifms_cols(fms_dt channels[max_fms_size],
					fms_dt cols[dw_tile_h][max_padding],
					const int tile_index_in_h,
					const int tile_index_in_w,
					int absolute_offset,
					const int padding_top,
					const int padding_left,
					const int padding_right,
					const int ifms_height,
					const int ifms_width,
					const fms_dt zero_point)
{
#pragma HLS INLINE

	const int starting_fill_w_offset = tile_index_in_w * dw_tile_w;

	for (int w = 0; w < max_padding; w++)
	{
#pragma HLS UNROLL
		if (w >= padding_right)
		{
			break;
		}
		for (int h = 0; h < dw_tile_h; h++)
		{
#pragma HLS UNROLL
			if (w + starting_fill_w_offset < ifms_width &&
				tile_index_in_w >= 0)
			{
				cols[h][w] = channels[absolute_offset + h * dw_tile_w + w];
			}
			else
			{
				cols[h][w] = zero_point;
			}
		}
	}
}

void fill_ifms_rows(fms_dt channels[max_fms_size],
					fms_dt rows[max_filter_hw_dim - 1][dw_tile_w],
					const int tile_index_in_h,
					const int tile_index_in_w,
					int absolute_offset,
					const int padding_top,
					const int padding_bottom,
					const int padding_left,
					const int ifms_height,
					const int ifms_width,
					const fms_dt zero_point)
{
#pragma HLS INLINE

	const int starting_fill_h_offset = tile_index_in_h * dw_tile_h;
	for (int h = 0; h < max_padding; h++)
	{
#pragma HLS UNROLL
		if (h >= padding_bottom)
		{
			break;
		}
		for (int w = 0; w < dw_tile_w; w++)
		{
#pragma HLS UNROLL
			if (h + starting_fill_h_offset < ifms_height &&
				tile_index_in_h >= 0)
			{
				rows[h][w] =
					channels[absolute_offset + h * dw_tile_w + w];
			}
			else
			{
				rows[h][w] = zero_point;
			}
		}
	}
}

void fill_ifms_corner(fms_dt channels[max_fms_size],
					  fms_dt lower_right_corner[max_padding * max_padding],
					  const int tile_index_in_h,
					  const int tile_index_in_w,
					  int absolute_offset,
					  const int padding_left, const int padding_top,
					  const int padding_right, const int padding_bottom,
					  const int ifms_height,
					  const int ifms_width,
					  const fms_dt zero_point)
{
#pragma HLS INLINE

	const int starting_fill_w_offset = tile_index_in_w * dw_tile_w;
	const int starting_fill_h_offset = tile_index_in_h * dw_tile_w;

	for (int h = 0; h < padding_bottom; h++)
	{
#pragma HLS UNROLL
		for (int w = 0; w < padding_right; w++)
		{
#pragma HLS UNROLL
			if (h + starting_fill_h_offset < ifms_height &&
				tile_index_in_h >= 0 &&
				w + starting_fill_w_offset < ifms_width &&
				tile_index_in_w >= 0)
			{
				lower_right_corner[h * +w] =
					channels[absolute_offset + h * (max_filter_hw_dim - 1) + w];
			}
			else
			{
				lower_right_corner[h * +w] = zero_point;
			}
		}
	}
}

void dw_fill_channels_tile(fms_dt channels[max_fms_size],
						   fms_dt channels_tile[pw_tile_h][pw_tile_w], const int starting_index,
						   int starting_d, const int layer_conv_d)
{
#pragma HLS INLINE

	for (int t_h = 0; t_h < pw_tile_h; t_h++)
	{
#pragma HLS PIPELINE
		for (int t_w = 0; t_w < pw_tile_w; t_w++)
		{
#pragma HLS UNROLL
			channels_tile[t_h][t_w] = channels[starting_index + t_h * pw_tile_w + t_w];
		}
	}
}

void dw_conv_fill_from_channels(fms_dt channels[max_fms_size],
								fms_dt ifm_tile[dw_tile_h][dw_tile_w],
								fms_dt padding_top_buffer[max_padding][dw_tile_w],
								fms_dt padding_right_buffer[dw_tile_h][max_padding],
								fms_dt padding_bottom_buffer[max_padding][dw_tile_w],
								fms_dt padding_left_buffer[dw_tile_h][max_padding],
								fms_dt padding_tl_corner[max_padding * max_padding],
								fms_dt padding_tr_corner[max_padding * max_padding],
								fms_dt padding_br_corner[max_padding * max_padding],
								fms_dt padding_bl_corner[max_padding * max_padding],
								const int ifm_width, const int ifm_height,
								const int ifms_depth,
								const int absolute_tile_offset_in_ifms,
								int ifm_tile_in_h, int ifm_tile_in_w,
								int tile_offset_in_d,
								const int num_of_tiles_w,
								const int padding_left, const int padding_top,
								const int padding_right, const int padding_bottom,
								const fms_dt fms_zero_point)
{
#pragma HLS INLINE off

	const int absolute_offset_padding_top = absolute_tile_offset_in_ifms - num_of_tiles_w * dw_tile_size + ((dw_tile_h - padding_top) * dw_tile_w);
	const int absolute_offset_padding_right = absolute_tile_offset_in_ifms + dw_tile_size;
	const int absolute_offset_padding_bottom = absolute_tile_offset_in_ifms + num_of_tiles_w * dw_tile_size;
	const int absolute_offset_padding_left = absolute_tile_offset_in_ifms - dw_tile_size + (dw_tile_w - padding_left);

	const int absolute_offset_padding_tl_corner = absolute_offset_padding_top - dw_tile_size + dw_tile_w - padding_left;
	const int absolute_offset_padding_tr_corner = absolute_offset_padding_top + dw_tile_size;
	const int absolute_offset_padding_br_corner = absolute_offset_padding_bottom + dw_tile_size;
	const int absolute_offset_padding_bl_corner = absolute_offset_padding_bottom - dw_tile_size + dw_tile_w - padding_left;

	if (padding_top > 0)
	{
		fill_ifms_rows(channels, padding_top_buffer, ifm_tile_in_h - 1, ifm_tile_in_w, absolute_offset_padding_top, padding_top, padding_bottom,
					   padding_left, ifm_height,
					   ifm_width, fms_zero_point);
	}
	fill_ifms_cols(channels, padding_right_buffer, ifm_tile_in_h, ifm_tile_in_w + 1, absolute_offset_padding_right, padding_top, padding_left, padding_right,
				   ifm_height, ifm_width, fms_zero_point);
	fill_ifms_rows(channels, padding_bottom_buffer, ifm_tile_in_h + 1, ifm_tile_in_w, absolute_offset_padding_bottom, padding_top, padding_bottom,
				   padding_left, ifm_height,
				   ifm_width, fms_zero_point);
	if (padding_left > 0)
	{
		fill_ifms_cols(channels, padding_left_buffer, ifm_tile_in_h, ifm_tile_in_w - 1, absolute_offset_padding_left, padding_top, padding_left, padding_right,
					   ifm_height, ifm_width, fms_zero_point);
	}

	if (padding_top > 0)
	{
		fill_ifms_corner(channels, padding_tl_corner, ifm_tile_in_h - 1, ifm_tile_in_w - 1, absolute_offset_padding_tl_corner,
						 padding_left, padding_top, padding_right, padding_bottom, ifm_height, ifm_width, fms_zero_point);
		fill_ifms_corner(channels, padding_tr_corner, ifm_tile_in_h - 1, ifm_tile_in_w + 1, absolute_offset_padding_tr_corner,
						 padding_left, padding_top, padding_right, padding_bottom, ifm_height, ifm_width, fms_zero_point);
	}
	fill_ifms_corner(channels, padding_br_corner, ifm_tile_in_h + 1, ifm_tile_in_w + 1, absolute_offset_padding_br_corner,
					 padding_left, padding_top, padding_right, padding_bottom, ifm_height, ifm_width, fms_zero_point);
	if (padding_left > 0)
	{
		fill_ifms_corner(channels, padding_bl_corner, ifm_tile_in_h + 1, ifm_tile_in_w - 1, absolute_offset_padding_bl_corner,
						 padding_left, padding_top, padding_right, padding_bottom, ifm_height, ifm_width, fms_zero_point);
	}
	dw_fill_channels_tile(channels, ifm_tile, absolute_tile_offset_in_ifms, tile_offset_in_d, ifms_depth);
}

void fill_dw_tile(fms_dt ifms_buffer[dw_max_v2_buffer_height][dw_max_v2_buffer_width],
				  fms_dt padding_top_buffer[max_padding][dw_tile_w],
				  fms_dt padding_right_buffer[dw_tile_h][max_padding],
				  fms_dt padding_bottom_buffer[max_padding][dw_tile_w],
				  fms_dt padding_left_buffer[dw_tile_h][max_padding],
				  fms_dt padding_tl_corner[max_padding * max_padding],
				  fms_dt padding_tr_corner[max_padding * max_padding],
				  fms_dt padding_br_corner[max_padding * max_padding],
				  fms_dt padding_bl_corner[max_padding * max_padding],
				  fms_dt ifm_tile[dw_tile_h][dw_tile_w],
				  const int padding_top, const int padding_left,
				  const int padding_bottom, const int padding_right,
				  int tile_in_h, int tile_in_w,
				  int absolute_offset_in_channels,
				  fms_dt fms_zero_point)
{
#pragma HLS INLINE off

	//***************************************
	for (int h = 0; h < max_padding; h++)
	{
#pragma HLS UNROLL
		if (h >= padding_top)
		{
			break;
		}
		for (int w = 0; w < max_padding; w++)
		{
#pragma HLS UNROLL
			if (w >= padding_left)
			{
				break;
			}
			ifms_buffer[h][w] = padding_tl_corner[h * padding_left + w];
		}
		for (int w = 0; w < max_padding; w++)
		{
#pragma HLS UNROLL
			if (w >= padding_right)
			{
				break;
			}
			ifms_buffer[h][w + dw_tile_w + padding_left] = padding_tr_corner[h * padding_left + w];
		}
	}

	for (int h = 0; h < max_padding; h++)
	{
		if (h >= padding_bottom)
		{
			break;
		}
		for (int w = 0; w < max_padding; w++)
		{
#pragma HLS UNROLL
			if (w >= padding_right)
			{
				break;
			}
			ifms_buffer[h + dw_tile_h + padding_top][w + dw_tile_w + padding_left] = padding_br_corner[h * padding_left + w];
		}
		for (int w = 0; w < max_padding; w++)
		{
#pragma HLS UNROLL
			if (w >= padding_left)
			{
				break;
			}
			ifms_buffer[h + dw_tile_h + padding_top][w] = padding_bl_corner[h * padding_left + w];
		}
	}
	//***************************************
	//***************************************
	if (padding_top > 0)
	{
		for (int h = 0; h < max_padding; h++)
		{
#pragma HLS UNROLL
			if (h >= padding_top)
			{
				break;
			}
			for (int w = 0; w < dw_tile_w; w++)
			{
#pragma HLS UNROLL
				ifms_buffer[h][w + padding_left] = padding_top_buffer[h][w];
			}
		}
	}

	for (int w = 0; w < max_padding; w++)
	{
#pragma HLS UNROLL
		if (w >= padding_right)
		{
			break;
		}
		for (int h = 0; h < dw_tile_h; h++)
		{
#pragma HLS UNROLL
			ifms_buffer[h + padding_top][w + dw_tile_w + padding_left] = padding_right_buffer[h][w];
		}
	}

	for (int h = 0; h < max_padding; h++)
	{
#pragma HLS UNROLL
		if (h >= padding_bottom)
		{
			break;
		}
		for (int w = 0; w < dw_tile_w; w++)
		{
#pragma HLS UNROLL
			ifms_buffer[h + dw_tile_h + padding_top][w + padding_left] = padding_bottom_buffer[h][w];
		}
	}

	if (padding_left > 0)
	{
		for (int w = 0; w < max_padding; w++)
		{
#pragma HLS UNROLL
			if (w >= padding_left)
			{
				break;
			}
			for (int h = 0; h < dw_tile_h; h++)
			{
#pragma HLS UNROLL
				ifms_buffer[h + padding_top][w] = padding_left_buffer[h][w];
			}
		}
	}
	//***************************************
	for (int h = 0; h < dw_tile_h; h++)
	{
#pragma HLS UNROLL
		for (int w = 0; w < dw_tile_w; w++)
		{
#pragma HLS UNROLL
			ifms_buffer[h + padding_top][w + padding_left] =
				ifm_tile[h][w];
		}
	}
}

void dw_conv_engine(dw_weights_dt weights[][max_filter_hw_dim * max_filter_hw_dim],
					fms_dt ifms_buffer[dw_max_v2_buffer_height][dw_max_v2_buffer_width],
					dw_pss_dt result_tile[dw_tile_h][dw_tile_w], const int filter_dim, const int strides)
{
#pragma HLS INLINE off
	for (int c_h = 0; c_h < max_filter_hw_dim; c_h++)
	{
		for (int c_w = 0; c_w < max_filter_hw_dim; c_w++)
		{
#pragma HLS PIPELINE
			for (int h = 0; h < dw_tile_h; h++)
			{
#pragma HLS UNROLL
				for (int w = 0; w < dw_tile_w; w++)
				{
#pragma HLS UNROLL
					if (c_h == 0 && c_w == 0)
					{
						result_tile[h][w] = 0;
					}
					if (c_w < filter_dim && c_h < filter_dim)
					{
						result_tile[h][w] += ifms_buffer[h + c_h][w + c_w] * weights[0][c_h * filter_dim + c_w]; // TODO
					}
				}
			}
		}
	}
}

void normalize_dw_result_tile(dw_pss_dt result_tile[dw_tile_h][dw_tile_w],
							  fms_dt normalized_tile[dw_tile_h][dw_tile_w],
							  fms_quantization_scheme normalization, const int layer_relu,
							  const fused_scales_dt fused_scales_tile[],
							  const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_tile[],
							  const relu_6_fused_scales_dt relu_6_fused_scales_tile[],
							  const biases_dt fused_zero_points_tile[])
{
#pragma HLS INLINE off

	normalization.fused_scales = fused_scales_tile[0];
	normalization.fused_scales_log_2_shift =
		fused_scales_log_2_shifts_tile[0];
	normalization.relu_6_fused_scale = relu_6_fused_scales_tile[0];
	normalization.fused_zero_point = fused_zero_points_tile[0];

	for (int h = 0; h < dw_tile_h; h++)
	{
#pragma HLS pipeline
		for (int w = 0; w < dw_tile_w; w++)
		{
#pragma HLS unroll
			normalized_tile[h][w] = dw_relu_norm(result_tile[h][w], normalization, layer_relu);
		}
	}
}

void normalize_and_write_back_result_tile(fms_dt result[max_fms_size], pss_dt pss_tile[dw_tile_h][dw_tile_w],
										  fms_quantization_scheme normalization, const int layer_relu,
										  const fused_scales_dt fused_scales_tile[],
										  const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_tile[],
										  const relu_6_fused_scales_dt relu_6_fused_scales_tile[],
										  const biases_dt fused_zero_points_tile[],
										  int absolute_offset_in_results)
{
#pragma HLS INLINE off

	normalization.fused_scales = fused_scales_tile[0];
	normalization.fused_scales_log_2_shift =
		fused_scales_log_2_shifts_tile[0];
	normalization.relu_6_fused_scale = relu_6_fused_scales_tile[0];
	normalization.fused_zero_point = fused_zero_points_tile[0];

	for (int h = 0; h < dw_tile_h; h++)
	{
#pragma HLS PIPELINE
		for (int w = 0; w < dw_tile_w; w++)
		{
#pragma HLS unroll
			result[absolute_offset_in_results + h * dw_tile_w + w] = dw_relu_norm(pss_tile[h][w], normalization, layer_relu);
		}
	}
}

void fill_dw_weights_and_scales_tiles(const dw_weights_dt weights[][3 * 3],
									  dw_weights_dt weights_tile[dw_tile_d][3 * 3],
									  const fused_scales_dt fused_scales[],
									  fused_scales_dt fused_scales_tile[],
									  const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
									  fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_tile[],
									  const relu_6_fused_scales_dt relu_6_fused_scales[],
									  relu_6_fused_scales_dt relu_6_fused_scales_tile[],
									  const biases_dt fused_zero_points[], biases_dt fused_zero_points_tile[],
									  int starting_d, const int current_dw_layer_weights_offset,
									  const int current_layer_fused_parameters_offset)
{
#pragma HLS INLINE off

	const int absolute_current_layer_fused_parameters_offset =
		current_layer_fused_parameters_offset + starting_d;
	const int absolute_current_layer_weights_offset = current_dw_layer_weights_offset + starting_d;
	for (int d = 0; d < dw_tile_d; d++)
	{
#pragma HLS PIPELINE
		for (int i = 0; i < 3 * 3; i++)
		{
#pragma HLS UNROLL
			weights_tile[d][i] = weights[absolute_current_layer_weights_offset + d][i];
		}
		fused_scales_tile[d] =
			fused_scales[absolute_current_layer_fused_parameters_offset + d];
		fused_scales_log_2_shifts_tile[d] =
			fused_scales_log_2_shifts[absolute_current_layer_fused_parameters_offset + d];
		relu_6_fused_scales_tile[d] =
			relu_6_fused_scales[absolute_current_layer_fused_parameters_offset + d];
		fused_zero_points_tile[d] =
			fused_zero_points[absolute_current_layer_fused_parameters_offset + d];
	}
}

void dw_fill_ofms_pss_tile(
	pss_dt src_pss_tile[dw_tile_h][dw_tile_w],
	pss_dt dst_pss_tile[dw_tile_h][dw_tile_w], const int offset_h, const int offset_w, const int strides)
{
#pragma HLS INLINE off
#pragma HLS PIPELINE

	for (int t_h = 0; t_h < dw_tile_h; t_h++)
	{
#pragma HLS UNROLL
		for (int t_w = 0; t_w < dw_tile_w; t_w++)
		{
#pragma HLS UNROLL
			if (t_h >= dw_tile_h / strides || t_w >= dw_tile_w / strides)
			{
				break;
			}
			dst_pss_tile[offset_h + t_h][offset_w + t_w] = src_pss_tile[t_h * strides][t_w * strides];
		}
	}
}

void dw_conv_3x3(const dw_weights_dt weights[][3 * 3], fms_dt channels[max_fms_size],
				 fms_dt result[max_fms_size], const int layer, const int layer_conv_d,
				 const int layer_ifm_width, const int layer_ifm_height,
				 const int num_of_tiles_d, const int num_of_ofms_tiles_h,
				 const int num_of_ofms_tiles_w, const int strides,
				 const int padding_left, const int padding_right, const int padding_top,
				 const int direction, const fused_scales_dt fused_scales[],
				 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
				 const relu_6_fused_scales_dt relu_6_fused_scales[],
				 const biases_dt fused_zero_points[])
{
#pragma HLS INLINE off

	const int padding_bottom = padding_right;
	const int num_of_ifms_tiles_h =
		(layer_ifm_height % dw_tile_h) == 0 ? layer_ifm_height / dw_tile_h : num_of_ofms_tiles_h * strides;
	const int num_of_ifms_tiles_w =
		(layer_ifm_width % dw_tile_w) == 0 ? layer_ifm_width / dw_tile_w : num_of_ofms_tiles_w * strides;

	const int num_of_ifms_tiles_hw = num_of_ifms_tiles_h * num_of_ifms_tiles_w;
	const int num_of_ofms_tiles_hw = num_of_ofms_tiles_h * num_of_ofms_tiles_w;

	fms_quantization_scheme normalization = {0, 0, 0, 0};

	const int current_dw_layer_weights_offset = dw_layers_weights_offsets[layer];
	const int current_layer_fused_parameters_offset =
		layers_fused_parameters_offsets[layer];

	dw_weights_dt weights_tile[dw_tile_d][3 * 3];
	//#pragma HLS ARRAY_PARTITION variable=weights_tile type=complete dim = 0

	fused_scales_dt fused_scales_tile[dw_tile_d];
	fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_tile[dw_tile_d];
	relu_6_fused_scales_dt relu_6_fused_scales_tile[dw_tile_d];
	biases_dt fused_zero_points_tile[dw_tile_d];

	normalization.ofm_zero_point = conv_fms_zero_points[layer + 1];
	normalization.ofm_scale_rec = conv_fms_scales_rec[layer + 1];
	normalization.ofm_scale = conv_fms_scales[layer + 1];

	const fms_dt current_layer_fms_zero_point = conv_fms_zero_points[layer];

	fms_dt ifm_tile[dw_tile_h][dw_tile_w];
	fms_dt ifms_buffer[dw_max_v2_buffer_height][dw_max_v2_buffer_width];
	fms_dt padding_top_buffer[max_padding][dw_tile_w];
	fms_dt padding_right_buffer[dw_tile_h][max_padding];
	fms_dt padding_bottom_buffer[max_padding][dw_tile_w];
	fms_dt padding_left_buffer[dw_tile_h][max_padding];
	fms_dt padding_tl_corner[max_padding * max_padding];
	fms_dt padding_tr_corner[max_padding * max_padding];
	fms_dt padding_bl_corner[max_padding * max_padding];
	fms_dt padding_br_corner[max_padding * max_padding];
	dw_pss_dt engine_result_tile[dw_tile_h][dw_tile_w];
	dw_pss_dt result_tile[dw_tile_h][dw_tile_w];

#pragma HLS ARRAY_PARTITION variable = ifm_tile type = complete dim = 0
#pragma HLS ARRAY_PARTITION variable = ifms_buffer type = complete dim = 0
#pragma HLS ARRAY_PARTITION variable = engine_result_tile type = complete dim = 0
#pragma HLS ARRAY_PARTITION variable = result_tile type = complete dim = 0

	for (int tile_in_d = 0; tile_in_d < num_of_tiles_d; tile_in_d++)
	{
		fill_dw_weights_and_scales_tiles(weights, weights_tile, fused_scales,
										 fused_scales_tile, fused_scales_log_2_shifts,
										 fused_scales_log_2_shifts_tile, relu_6_fused_scales,
										 relu_6_fused_scales_tile, fused_zero_points,
										 fused_zero_points_tile, tile_in_d * dw_tile_d,
										 current_dw_layer_weights_offset,
										 current_layer_fused_parameters_offset);

		for (int ofm_tile_in_h = 0; ofm_tile_in_h < num_of_ofms_tiles_h;
			 ofm_tile_in_h++)
		{
			int absolute_offset_in_ofms = tile_in_d * num_of_ofms_tiles_hw * dw_tile_size + ofm_tile_in_h * num_of_ofms_tiles_w * dw_tile_size;
			for (int ifm_tile_in_ofm_tile_h = 0;
				 ifm_tile_in_ofm_tile_h < strides;
				 ifm_tile_in_ofm_tile_h++)
			{
				int ifm_tile_in_h = ofm_tile_in_h * strides + ifm_tile_in_ofm_tile_h;
				int absolute_offset_in_ifms = tile_in_d * num_of_ifms_tiles_hw * dw_tile_size +
											  (ofm_tile_in_h * strides + ifm_tile_in_ofm_tile_h) * num_of_ifms_tiles_w * dw_tile_size;

				for (int ofm_tile_in_w = 0; ofm_tile_in_w < num_of_ofms_tiles_w;
					 ofm_tile_in_w++)
				{
					for (int ifm_tile_in_ofm_tile_w = 0;
						 ifm_tile_in_ofm_tile_w < strides;
						 ifm_tile_in_ofm_tile_w++)
					{
						int ifm_tile_in_w = ofm_tile_in_w * strides + ifm_tile_in_ofm_tile_w;
						//*************************
						dw_conv_fill_from_channels(channels, ifm_tile,
												   padding_top_buffer, padding_right_buffer, padding_bottom_buffer, padding_left_buffer,
												   padding_tl_corner, padding_tr_corner, padding_br_corner, padding_bl_corner,
												   layer_ifm_width, layer_ifm_height, layer_conv_d,
												   absolute_offset_in_ifms,
												   ifm_tile_in_h, ifm_tile_in_w,
												   tile_in_d * dw_tile_d,
												   num_of_ifms_tiles_w,
												   padding_left, padding_top, padding_right, padding_bottom,
												   current_layer_fms_zero_point);
						fill_dw_tile(ifms_buffer,
									 padding_top_buffer, padding_right_buffer, padding_bottom_buffer, padding_left_buffer,
									 padding_tl_corner, padding_tr_corner, padding_br_corner, padding_bl_corner,
									 ifm_tile, padding_top, padding_left, padding_bottom, padding_right, ifm_tile_in_h, ifm_tile_in_w,
									 absolute_offset_in_ifms, current_layer_fms_zero_point);
						dw_conv_engine(weights_tile, ifms_buffer, engine_result_tile, 3, strides);
						if(layer == 10)
						dw_fill_ofms_pss_tile(engine_result_tile, result_tile, ifm_tile_in_ofm_tile_h * dw_tile_h / strides, ifm_tile_in_ofm_tile_w * dw_tile_w / strides, strides);
						// normalize_dw_result_tile(result_tile, normalized_tile, normalization, 6, fused_scales_tile, fused_scales_log_2_shifts_tile,
						// 						 relu_6_fused_scales_tile, fused_zero_points_tile, tile_in_d);
						// write_back_result_tile(result, normalized_tile, absolute_offset_in_ofms + (ofm_tile_in_w * dw_tile_size) + ifm_tile_in_ofm_tile_h * dw_tile_size / strides, ifm_tile_in_ofm_tile_w * dw_tile_w / strides, strides);

						// if (tile_in_d == 0 && ifm_tile_in_h == 0 && ifm_tile_in_w == 1 && layer == 7)
						// {
						// 	cout << "\n****************\n"
						// 		 << "fff"
						// 		 << "\n******************\n";
						// 	cout << "\n"
						// 		 << (int)current_layer_fms_zero_point << "\n";
						// 	cout << "\n"
						// 		 << (int)fused_scales_tile[0] << "\n";
						// 	cout << "\n"
						// 		 << (int)fused_scales_log_2_shifts_tile[0] << "\n";
						// 	cout << "\n"
						// 		 << (int)relu_6_fused_scales_tile[0] << "\n";
						// 	cout << "\n"
						// 		 << (int)fused_zero_points_tile[0] << "\n";
						// 	for (int h = 0; h < dw_max_v2_buffer_height; h++)
						// 	{
						// 		for (int w = 0; w < dw_max_v2_buffer_width; w++)
						// 		{
						// 			cout << (int)ifms_buffer[h][w] << " ";
						// 		}
						// 		cout << "\n";
						// 	}
						// 	cout << "\n************\n";
						// 	cout<<(int)padding_bl_corner[0]<<"\n";
						// 	cout<<(int)padding_br_corner[0]<<"\n";
						// 	cout << "\n";
						// 	for (int i = 0; i < dw_tile_w; i++)
						// 	{
						// 		cout << (int)padding_left_buffer[i][0] << " ";
						// 	}
						// 	cout << "\n";
						// 	// for (int h = 0; h < dw_tile_h; h++)
						// 	// {
						// 	// 	for (int w = 0; w < dw_tile_w; w++)
						// 	// 	{
						// 	// 		cout << (int)normalized_tile[h][w] << " ";
						// 	// 	}
						// 	// 	cout << "\n";
						// 	// }
						// }
						//**********************
						absolute_offset_in_ifms += dw_tile_size;
					}
				}
				normalize_and_write_back_result_tile(result, result_tile, normalization, 6, fused_scales_tile, fused_scales_log_2_shifts_tile,
													 relu_6_fused_scales_tile, fused_zero_points_tile, absolute_offset_in_ofms);
			}
		}
	}
}
