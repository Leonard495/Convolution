#pragma once

enum {FILTER_X = 11, FILTER_Y = 11};

#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024

#define BLOCK_SIZE 32

#define B (2)

#define M(x) (((x)-1)/(B) + 1)
#define REG_WIDTH (M(FILTER_X+B-1)*B)

#if(B == 32)
typedef uint16 bus_t;
#elif(B == 16)
typedef uint8 bus_t;
#elif(B == 8)
typedef uint4 bus_t;
#elif(B == 4)
typedef uint2 bus_t;
#elif(B == 2)
typedef uint bus_t;
#endif

void filter_image_global(short* img, short* filter,
					size_t width, size_t height,
					short* output,
					int max_threads_dim[3]);

void filter_image_shared(short* img, short* filter,
					size_t width, size_t height,
					short* output,
					int max_threads_dim[3]);
