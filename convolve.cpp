// Convolve Example

#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <unistd.h>

// CUDA includes
#include <cuda_runtime.h>
#include <driver_types.h>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/timer/timer.hpp>

#include "exceptions.h"
#include <convolve_kernel.h>
#include <my_log.h>

namespace po = boost::program_options;

void readXFile(std::string filename, size_t rows, size_t cols, short* output_buf)
{
	std::ifstream txtFile(filename.c_str());
	if(!txtFile.is_open())
	{
		std::cout << "ERROR: Couldn't open file" << filename << std::endl;
		abort();
	}

	if (!output_buf)
		abort();

	for (size_t r = 0; r < cols; r++) {
		for (size_t c = 0; c < cols; c++) {
			txtFile >> output_buf[r * rows + c]; //mat.at<short>(r, c);
		}
	}
}

int main(int argc, char* argv[])
{
	enum {IMAGE_X = 1024, IMAGE_Y = 1024};

	try
	{
		po::options_description desc("Allowed options");

		desc.add_options()
				("help", "produce help message")
				("duration,D", po::value<unsigned int>()->default_value(30), "test duration in seconds")
				("image,I", po::value<std::string>(), "input image file name")
				("filter,F", po::value<std::string>(), "filter file name")
				("width,W", po::value<unsigned long>()->default_value(IMAGE_X), "image width")
				("height,S", po::value<unsigned long>()->default_value(IMAGE_Y), "image height")
				("kernel,K", po::value<std::string>()->default_value("CUDA_SHARED"), "accelerator type may be CUDA_SHARED or CUDA_GLOBAL");

		po::variables_map vars;
		po::store(po::parse_command_line(argc, argv, desc), vars);

		if(vars.size() <= 1 || vars.count("help"))
		{
			std::cout << desc << std::endl;
			return 1;
		}

		if(!vars.count("image"))
			THROW("'image' parameter is not set");
		if(!vars.count("filter"))
			THROW("'filter' parameter is not set");

		auto test_duration = vars["duration"].as<unsigned int>();
		auto image_filename = vars["image"].as<std::string>();
		auto filter_filename = vars["filter"].as<std::string>();
		auto im_width = vars["width"].as<unsigned long>();
		auto im_height = vars["height"].as<unsigned long>();
		auto device_type = vars["kernel"].as<std::string>();

		std::cout << "input image file " << image_filename.c_str() << std::endl;
		std::cout << "input filter file " << filter_filename.c_str() << std::endl;

		std::cout << "Connecting to device....." << std::endl;

		if (device_type == "CUDA_SHARED") {
			cudaError_t err;
			short* input_img_cuda_ptr;
			short* flt_img_cuda_ptr;
			short* output_img_cuda_ptr;
			size_t img_size = 0;
			size_t filter_size = 0;
			int dev_number = 0;
			struct cudaDeviceProp cuda_dev_prop;
			cudaEvent_t write_start, write_stop, process_start, process_stop, read_start, read_stop;

			cudaEventCreate(&write_start);
			cudaEventCreate(&write_stop);
			cudaEventCreate(&process_start);
			cudaEventCreate(&process_stop);
			cudaEventCreate(&read_start);
			cudaEventCreate(&read_stop);

			// Input image
//	    	cv::Mat input_img_s16(1024, 1024, CV_16S);
//	    	cv::Mat input_flt_s16(FILTER_X, FILTER_Y, CV_16S);

	    	img_size = im_width * im_height * sizeof(short);
	    	filter_size = FILTER_X * FILTER_Y * sizeof(short);
	    	std::cout << "Image size = " << img_size << std::endl;
	    	std::cout << "Filter size = " << filter_size << std::endl;

	    	std::vector<short> image_buf(img_size);
	    	std::vector<short> filter_buf(filter_size);
	    	std::vector<short> output_buf(img_size);

	    	{
	    		std::ifstream f_in(image_filename, std::ios::binary);
	    		if (f_in.is_open())
	    			f_in.read((char*) &image_buf[0], im_width * im_height * sizeof(short));
	    		else
	    		{
	    			MSG("Unable to open file %s", image_filename.c_str());
	    			THROW("Error");
	    		}
	    	}

	    	// Input filter
	    	{
	    		std::ifstream f_in(filter_filename, std::ios::binary);
	    		if (f_in.is_open())
	    			f_in.read((char*) &filter_buf[0], FILTER_X * FILTER_Y * sizeof(short));
	    		else
	    		{
	    			MSG("Unable to open file %s", filter_filename.c_str());
	    			THROW("Error");
	    		}
	    	}
#ifdef USE_HIGH_FREQUENCY_FILTER
	    	for (int row = 0; row < FILTER_Y; ++row)
	    	{
	    		for (int col = 0; col < FILTER_X; ++col)
	    		{
	    			int idx = row * FILTER_Y + col;
	    			if (FILTER_X * FILTER_Y / 2 == idx)
	    				filter_buf[idx] = FILTER_X * FILTER_Y - 1;
	    			else
	    				filter_buf[idx] = -1;
	    			LOG_MSG("%d, ", filter_buf[idx]);
	    		}
	    		LOG_MSG("\n");
	    	}
	    	{
	    		char a;
	    		std::cin >> a;
	    	}
#endif
	    	// ------- Explore CUDA device -------
	    	err = cudaGetDeviceCount(&dev_number);
	    	if (err)
	    		THROW( (std::string("CUDA malloc input image: ") + (char*) cudaGetErrorString(err)).c_str() );
	    	std::cout << "CUDA devices available - " << dev_number << std::endl;

	    	err = cudaGetDeviceProperties(&cuda_dev_prop, 0);
	    	if (err)
	    		THROW( (std::string("CUDA malloc input image: ") + (char*) cudaGetErrorString(err)).c_str() );

	    	std::cout << "CUDA device name: "				<< cuda_dev_prop.name				<< std::endl;
	    	std::cout << "CUDA device total global mem: "	<< cuda_dev_prop.totalGlobalMem		<< std::endl;
	    	std::cout << "CUDA device sharedMemPerBlock: "	<< cuda_dev_prop.sharedMemPerBlock	<< std::endl;
	    	std::cout << "CUDA device maxThreadsPerBlock: "	<< cuda_dev_prop.maxThreadsPerBlock	<< std::endl;
	    	std::cout << "CUDA device maxThreadsDim[0]: "	<< cuda_dev_prop.maxThreadsDim[0]	<< std::endl;
	    	std::cout << "CUDA device maxThreadsDim[1]: "	<< cuda_dev_prop.maxThreadsDim[1]	<< std::endl;
	    	std::cout << "CUDA device maxThreadsDim[2]: "	<< cuda_dev_prop.maxThreadsDim[2]	<< std::endl;
	    	std::cout << "CUDA device maxGridSize[0]: "		<< cuda_dev_prop.maxGridSize[0]		<< std::endl;
	    	std::cout << "CUDA device maxGridSize[1]: "		<< cuda_dev_prop.maxGridSize[1]		<< std::endl;
	    	std::cout << "CUDA device maxGridSize[2]: "		<< cuda_dev_prop.maxGridSize[2]		<< std::endl;
	    	std::cout << "CUDA device pciBusID: "			<< cuda_dev_prop.pciBusID			<< std::endl;

	    	// -----------------------------------

	    	// ----- Allocate memory on the GPU -----
	    	err = cudaMalloc(&input_img_cuda_ptr, img_size);
	    	if (err)
	    	{
	    		std::string reason;
	    		reason = std::string("CUDA malloc input image: ") + (char*) cudaGetErrorString(err);
	    		THROW(reason.c_str());
	    	}

	    	err = cudaMalloc(&flt_img_cuda_ptr, filter_size);
	    	if (err)
	    	{
	    		std::string reason;
	    		reason = std::string("CUDA malloc filter image: ") + (char*) cudaGetErrorString(err);
	    		THROW(reason.c_str());
	    	}

	    	err = cudaMalloc(&output_img_cuda_ptr, img_size);
	    	if (err)
	    	{
	    		std::string reason;
	    		reason = std::string("CUDA malloc input image: ") + (char*) cudaGetErrorString(err);
	    		THROW(reason.c_str());
	    	}
	    	// --------------------------------------

	    	// Clear output image
	    	size_t len = output_buf.size();
	    	for (int i = 0; i < len; ++i)
	    		output_buf[i] = 0;

	    	cudaMemcpy(	output_img_cuda_ptr,			// Copy image from CPU to GPU
	    				&output_buf[0],
						len,
						cudaMemcpyHostToDevice);

	    	auto start = std::chrono::high_resolution_clock::now();
	    	cudaEventRecord(write_start, 0);

	    	// ----- Load data to GPU -----
	    	cudaMemcpy(	input_img_cuda_ptr,			// Copy image from CPU to GPU
	    				&image_buf[0],
						img_size,
						cudaMemcpyHostToDevice);
	    	cudaMemcpy(	flt_img_cuda_ptr,			// Copy filter from CPU to GPU
	    				&filter_buf[0],
						filter_size,
						cudaMemcpyHostToDevice);
	    	// ----------------------------
	    	cudaEventRecord(write_stop, 0);
	    	cudaEventSynchronize(write_stop);
	    	float write_elapsed_time;
	    	cudaEventElapsedTime(&write_elapsed_time, write_start, write_stop);

	    	// ----- Process image on the GPU -----
	    	{
	    		boost::timer::cpu_timer operation_time;
	    		int iterations = 0;
	    		float time_total = 0;

	    		while ((float) operation_time.elapsed().wall / 1000000 < test_duration * 1000)	// 30 sec
	    		{
					if (0 == iterations++ % 10)
						LOG_MSG("Iteration %d is in progress, test time %2.1f sec\n", iterations, (float) operation_time.elapsed().wall / (1000 * 1000 * 1000));

					if (device_type == "CUDA_SHARED")
					{
						cudaEventRecord(process_start, 0);
						filter_image_shared(input_img_cuda_ptr,
									 flt_img_cuda_ptr,
									 im_width,
									 im_height,
									 output_img_cuda_ptr,
									 cuda_dev_prop.maxThreadsDim);

						cudaEventRecord(process_stop, 0);
						cudaEventSynchronize(process_stop);
					}
					else if (device_type == "CUDA_GLOBAL")
					{
						cudaEventRecord(process_start, 0);
						filter_image_global(input_img_cuda_ptr,
									 flt_img_cuda_ptr,
									 im_width,
									 im_height,
									 output_img_cuda_ptr,
									 cuda_dev_prop.maxThreadsDim);

						cudaEventRecord(process_stop, 0);
						cudaEventSynchronize(process_stop);
					}
					float process_elapsed_time;
					cudaEventElapsedTime(&process_elapsed_time, process_start, process_stop);
					time_total += process_elapsed_time;
	    		}
	    		MSG("Single kernel execution time %2.1f ms", time_total / iterations);
	    	}
	    	// ------------------------------------

	    	// ----- Read result back to CPU ------
	    	cudaEventRecord(read_start, 0);
	    	cudaMemcpy(	&output_buf[0],					// Copy from GPU to CPU
	    				output_img_cuda_ptr,
						img_size,
						cudaMemcpyDeviceToHost);
	    	cudaEventRecord(read_stop, 0);
	    	cudaEventSynchronize(read_stop);
	    	float read_elapsed_time;
	    	cudaEventElapsedTime(&read_elapsed_time, read_start, read_stop);
	    	// ------------------------------------

			{
				std::ofstream f_out("output.raw", std::ios::binary);
				f_out.write((char*) &output_buf[0], im_width * im_height * sizeof(short));
			}

			auto finish = std::chrono::high_resolution_clock::now();
			auto duration_usecs = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();

			cudaFree(input_img_cuda_ptr);
			cudaFree(flt_img_cuda_ptr);
			cudaFree(output_img_cuda_ptr);

		    std::cout << "It took me " << write_elapsed_time << " ms to write data to GPU" << std::endl;
		    std::cout << "It took me " << read_elapsed_time << " ms to read data from GPU" << std::endl;

			cudaEventDestroy(write_start);
			cudaEventDestroy(write_stop);
			cudaEventDestroy(process_start);
			cudaEventDestroy(process_stop);
			cudaEventDestroy(read_start);
			cudaEventDestroy(read_stop);

		}
	}
	catch(const std::exception &e)
	{
		std::cout << "Fatal error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

    std::cout << "Completed Successfully" << std::endl;

    return EXIT_SUCCESS;
}

