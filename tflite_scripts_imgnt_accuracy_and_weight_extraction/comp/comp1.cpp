const int starting_fill_h_offset = tile_index_in_h * dw_tile_h;
	const int starting_fill_w_offset = tile_index_in_w * dw_tile_w;
	for (int h = 0; h < max_filter_hw_dim - 1; h++)
	{
		for (int w = 0; w < dw_tile_w; w++)
		{
#pragma HLS UNROLL
			if (h + starting_fill_h_offset < ifms_height &&
				w + starting_fill_w_offset < ifms_width)
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