cmake_minimum_required(VERSION 3.24)
project(soft_ctc_gpu CXX)

IF(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
ENDIF()

option(ENABLE_PYBIND "PyBind library." ON)
option(ENABLE_COMPUTE_CACHE "Compute Cache" ON)
option(ENABLE_OPENCL "OpenCL" OFF)
option(ENABLE_CUDA "Cuda" ON)
option(ENABLE_TORCH "Torch" ON)
option(ENABLE_PROFILLING "Profilling" OFF)

file (GLOB H_FILES src/*.h src/*.hpp src/*.cuh)
file (GLOB CPP_FILES src/*.cpp src/*.c src/*.cxx)

IF(ENABLE_OPENCL)
	file (GLOB H_OPENCL_FILES src/OpenCL/*.h src/OpenCL/*.hpp)
	file (GLOB CPP_OPENCL_FILES src/OpenCL/*.cpp src/OpenCL/*.c src/OpenCL/*.cxx)
ENDIF()

IF(ENABLE_CUDA)
	file (GLOB H_CUDA_FILES src/Cuda/*.h src/Cuda/*.hpp src/Cuda/*.cuh)
	file (GLOB CPP_CUDA_FILES src/Cuda/*.cpp src/Cuda/*.c src/Cuda/*.cxx src/Cuda/*.cu)
	add_library(soft_ctc_cuda STATIC ${H_CUDA_FILES} ${CPP_CUDA_FILES} ${H_FILES})
ENDIF()

IF(ENABLE_PYBIND)
	find_package(pybind11 REQUIRED)
	pybind11_add_module(soft_ctc_gpu ${CPP_FILES} ${H_FILES} ${H_OPENCL_FILES} ${CPP_OPENCL_FILES} ${H_CUDA_FILES})
	
	target_compile_definitions(soft_ctc_gpu PUBLIC USE_PYBIND)
	
ELSE()
	add_executable(soft_ctc_gpu ${CPP_FILES} ${H_FILES} ${H_OPENCL_FILES} ${CPP_OPENCL_FILES} ${H_CUDA_FILES})
ENDIF()

set_target_properties(soft_ctc_gpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../soft_ctc/)
set_target_properties(soft_ctc_gpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/../soft_ctc/)
set_target_properties(soft_ctc_gpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/../soft_ctc/)

IF(ENABLE_OPENCL)
	find_package( OpenCL REQUIRED )
	include_directories(soft_ctc_gpu PUBLIC ${OpenCL_INCLUDE_DIRS})
	target_link_libraries(soft_ctc_gpu PUBLIC ${OpenCL_LIBRARIES})
	target_compile_definitions(soft_ctc_gpu PUBLIC USE_OPENCL)
ENDIF()

IF(ENABLE_CUDA)
  # enable_language(CUDA)
	target_compile_definitions(soft_ctc_gpu PUBLIC USE_CUDA)
  set_property(TARGET soft_ctc_cuda PROPERTY CUDA_ARCHITECTURES native)
ENDIF()

IF(ENABLE_TORCH)
	find_package(Torch REQUIRED)
	find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
	target_compile_definitions(soft_ctc_gpu PUBLIC USE_TORCH)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
	target_link_libraries(soft_ctc_gpu PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
	IF(ENABLE_CUDA)
		target_compile_definitions(soft_ctc_cuda PUBLIC USE_TORCH)
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CXX_FLAGS}")
		target_link_libraries(soft_ctc_cuda PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
	ENDIF()
	IF(MSVC)
	  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	  add_custom_command(TARGET soft_ctc_gpu
						 POST_BUILD
						 COMMAND ${CMAKE_COMMAND} -E copy_if_different
						 ${TORCH_DLLS}
						 $<TARGET_FILE_DIR:soft_ctc_gpu>)
	ENDIF(MSVC)
ENDIF()

set_property(TARGET soft_ctc_gpu PROPERTY CXX_STANDARD 17)
set_property(TARGET soft_ctc_cuda PROPERTY CXX_STANDARD 17)


IF(ENABLE_COMPUTE_CACHE)
	target_compile_definitions(soft_ctc_gpu PUBLIC COMPUTE_CACHE_ENABLED)
ENDIF()

IF(ENABLE_CUDA)
	target_link_libraries(soft_ctc_gpu PUBLIC soft_ctc_cuda)
	IF(NOT MSVC)
		target_compile_options(soft_ctc_cuda PUBLIC -fPIC)
    ENDIF()
ENDIF()

IF(ENABLE_PROFILLING)
	target_compile_definitions(soft_ctc_gpu PUBLIC USE_PROFILLING)
	IF(ENABLE_CUDA)
		target_compile_definitions(soft_ctc_cuda PUBLIC USE_PROFILLING)
	ENDIF()
ENDIF()
add_custom_command(TARGET soft_ctc_gpu POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:soft_ctc_gpu>)
