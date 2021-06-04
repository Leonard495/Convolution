cmake_minimum_required(VERSION 2.8)

project(convolve)

set(BOOST_ROOT /usr/lib)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)

set(CUDA_ROOT /usr/local/cuda-7.5/lib64)

find_package(Boost COMPONENTS program_options timer chrono system REQUIRED)
find_package(CUDA COMPONENTS cudart REQUIRED)

set(SOURCE
	convolve.cpp
   )

# Find image to be convolved
file(GLOB IMAGE_SRC "data/image.raw")
message ("Input image file - ${IMAGE_SRC}")
# Find convolution matrix
file(GLOB FILTER_SRC "data/filter.raw")
message ("Input filter file - ${FILTER_SRC}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} CUDA_INCLUDE_DIRS)

add_definitions(-std=c++11)

cuda_add_executable(
    convolve
    convolve_kernel.cu
    ${SOURCE})

target_link_libraries(convolve dl rt ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})
##########################################################################
# Create runs.sh
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/run.sh
	    COMMAND echo "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME} --image ${IMAGE_SRC} --filter ${FILTER_SRC} --kernel CUDA_SHARED" >> ${CMAKE_CURRENT_BINARY_DIR}/run.sh
	    COMMAND chmod +x ${CMAKE_CURRENT_BINARY_DIR}/run.sh
	    )
add_custom_target(generate_run_sh ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/run.sh)
