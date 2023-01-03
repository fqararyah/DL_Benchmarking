for (int w = 0; w < dw_tile_w; w++)
		{
#pragma HLS UNROLL
			if (w + padding_top_starting_fill_w_offset < ifms_width + padding_left &&
				padding_top_tile_index_in_h >= 0)
			{
				padding_top_buffer[h][w] =
					channels[padding_top_absolute_offset + h * dw_tile_w + w];
			}
			else
			{
				padding_top_buffer[h][w] = fms_zero_point;
			}
		}