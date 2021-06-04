#include <stdio.h>
#include <cstdlib>
#include "convolve_kernel.h"

enum {WH_UNROLL = 3};

__global__ void cuda_filter_image_global(short* input, short* coef_buf,
								   	 size_t width, size_t height,
									 short* output) {
	for (int y = threadIdx.y; y < height; y += blockDim.y) {

		short* output_line = (short*) &output[y * width];

		for (int x = threadIdx.x; x < width; x += blockDim.x) {

			// Convolution
			int sum = 0;
			for (int f_y = 0; f_y < FILTER_Y; f_y++) {
				short* flt_line = (short*) &coef_buf[f_y * FILTER_X];

				for (int f_x = 0; f_x < FILTER_X; f_x++) {
					int img_y = y + f_y - FILTER_Y / 2;
					int img_x = x + f_x - FILTER_X / 2;

					if ((img_y >= 0) && (img_y < height) &&
					    (img_x >= 0 ) && (img_x < width))
					{
						sum += (int) flt_line[f_x] * (int) input[img_y * width + img_x];
					}
				}
			}

			/* Handle Saturation */
			if (sum > SHRT_MAX) {
				sum = SHRT_MAX;
			} else if (sum < SHRT_MIN) {
				sum = SHRT_MIN;
			}

			output_line[x] = sum;
		}
	}
}

__global__ void cuda_filter_image_shared(short* input, short* coef_buf,
								   	 size_t width, size_t height,
									 short* output) {
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	int row = threadIdx.y;
	int col = threadIdx.x;

	// Select first element of sub-matrix
	__shared__ short image_loc[BLOCK_SIZE*WH_UNROLL][BLOCK_SIZE*WH_UNROLL];
	__shared__ int filter_loc[FILTER_Y][FILTER_X];

	// Copy filter to shared memory
	if ((row < FILTER_Y) && (col < FILTER_X))
		filter_loc[row][col] = coef_buf[row * FILTER_X + col];

	// Copy part of image to the shared memory.
	for (int row = threadIdx.y; row < threadIdx.y + BLOCK_SIZE*WH_UNROLL; row += BLOCK_SIZE) {
		for (int col = threadIdx.x; col < threadIdx.x + BLOCK_SIZE*WH_UNROLL; col += BLOCK_SIZE) {
			// Each thread reads one element
			if ((blockRow * (BLOCK_SIZE*WH_UNROLL - FILTER_Y) + row < height) &&
				(blockCol * (BLOCK_SIZE*WH_UNROLL - FILTER_X) + col < width) &&
				(blockRow * (BLOCK_SIZE*WH_UNROLL - FILTER_Y) + row - FILTER_Y / 2 >= 0) &&
				(blockCol * (BLOCK_SIZE*WH_UNROLL - FILTER_X) + col - FILTER_X / 2 >= 0))
			{	// Central part
				image_loc[row][col] = input[(blockRow * (BLOCK_SIZE*WH_UNROLL - FILTER_Y) + row - FILTER_Y / 2) * width +
					blockCol * (BLOCK_SIZE*WH_UNROLL - FILTER_X) + col - FILTER_X / 2];
			}
			else
			{	// Peripheral part
				image_loc[row][col] = 0;
			}
		}
	}

	__syncthreads();	// Filled shared memory

	for (int row = threadIdx.y; row < threadIdx.y + BLOCK_SIZE*WH_UNROLL; row += BLOCK_SIZE) {
		for (int col = threadIdx.x; col < threadIdx.x + BLOCK_SIZE*WH_UNROLL; col += BLOCK_SIZE) {
			if ((row >= FILTER_Y/2 ) && (row <= BLOCK_SIZE*WH_UNROLL - FILTER_Y/2 - 1) &&
				(col >= FILTER_X/2) && (col <= BLOCK_SIZE*WH_UNROLL - FILTER_X/2 - 1))
			{
				// Convolve
				int sum = 0;
				#pragma unroll FILTER_X
				for (int f_y = 0; f_y < FILTER_Y; f_y++)
				{
					#pragma unroll FILTER_X
					for (int f_x = 0; f_x < FILTER_X; f_x++)
					{
						sum += filter_loc[f_y][f_x] * image_loc[row + f_y - FILTER_Y / 2][col + f_x - FILTER_X / 2];
					}
				}
				// Handle Saturation
				if (sum > SHRT_MAX) {
					sum = SHRT_MAX;
				}
				else if (sum < SHRT_MIN) {
					sum = SHRT_MIN;
				}
				// Each thread writes one element
				output[(blockRow * (BLOCK_SIZE*WH_UNROLL - FILTER_Y) + row - FILTER_Y / 2) * width +
					blockCol * (BLOCK_SIZE*WH_UNROLL - FILTER_X) + col - FILTER_X / 2] = sum;
			}
		}
	}
}

void filter_image_global(short* img, short* filter,
					size_t width, size_t height,
					short* output,
					int max_threads_dim[3])
{
	dim3 dimBlock(sqrt(max_threads_dim[0]), sqrt(max_threads_dim[0])); 		//
	dim3 block_nmb(1, 1);	//
	cuda_filter_image_global <<< block_nmb, dimBlock >>> (img, filter, width, height, output);
}

void filter_image_shared(short* img, short* filter,
					size_t width, size_t height,
					short* output,
					int max_threads_dim[3])
{
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); //
	dim3 block_nmb(width / (WH_UNROLL*BLOCK_SIZE - FILTER_X), height / (WH_UNROLL*BLOCK_SIZE - FILTER_Y));
	cuda_filter_image_shared <<< block_nmb, dimBlock >>> (img, filter, width, height, output);
}

